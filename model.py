import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torch.nn.init import kaiming_normal_, constant_

from params import par
from IMU import IMUFeatureExtractor, VisualInertialFusion

def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, dropout=0):
    """Create a convolutional block with optional batch normalization."""
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)
        )

# Define a Basic Block for ResNet (used by MonoDepth2 encoder)
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# MonoDepth2 Encoder (ResNet-18)
class MonoDepthEncoder(nn.Module):
    def __init__(self, in_channels=2, num_features=512, pretrained_path=None):
        super(MonoDepthEncoder, self).__init__()
        self.inplanes = 64

        # Initial conv layer for depth (2 channels: consecutive frame pairs)
        self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers (standard ResNet-18 architecture to match MonoDepth2)
        self.layer1 = self._make_layer(BasicBlock, 64, 2)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        # Global average pooling and projection
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_features)

        # Load pretrained weights if provided
        if pretrained_path is not None:
            self._load_pretrained(pretrained_path)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _load_pretrained(self, pretrained_path):
        """Load pretrained MonoDepth2 encoder weights."""
        print(f"Loading pretrained MonoDepth2 encoder from: {pretrained_path}")
        pretrained_dict = torch.load(pretrained_path, map_location='cpu')
        
        # Print first few keys to understand the structure
        print(f"Original keys format: {list(pretrained_dict.keys())[:3]}")
        
        # Remove the 'encoder.' prefix from keys if present
        new_pretrained_dict = {}
        for k, v in pretrained_dict.items():
            if k.startswith('encoder.'):
                new_key = k[8:]  # Remove 'encoder.' prefix
                new_pretrained_dict[new_key] = v
            else:
                new_pretrained_dict[k] = v
        
        # Map MonoDepth2 encoder weights to our model
        model_dict = self.state_dict()
        new_dict = {}
        matched_params = 0
        total_params = sum(p.numel() for p in self.parameters())
        
        for k, v in new_pretrained_dict.items():
            # Adapt the input layer (MonoDepth2 expects 3 channels, we have 2)
            if k == 'conv1.weight':
                pretrained_weight = v  # Shape: [64, 3, 7, 7]
                new_weight = pretrained_weight.mean(dim=1, keepdim=True).repeat(1, 2, 1, 1)  # Shape: [64, 2, 7, 7]
                new_dict['conv1.weight'] = new_weight
                matched_params += new_weight.numel()
            elif k in model_dict and model_dict[k].shape == v.shape:
                new_dict[k] = v
                matched_params += v.numel()
            else:
                print(f"Skipping {k} due to shape mismatch or not found in model")
        
        model_dict.update(new_dict)
        self.load_state_dict(model_dict)
        print(f"Pretrained MonoDepth2 encoder loaded successfully")
        print(f"Initialized {matched_params/total_params*100:.2f}% of parameters from pretrained model")

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# Cross-Modal Attention for RGB and Depth Fusion
class CrossModalAttention(nn.Module):
    def __init__(self, feature_dim, num_heads=8):
        super(CrossModalAttention, self).__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        assert self.head_dim * num_heads == feature_dim, "Feature dimension must be divisible by number of heads"

        # Q, K, V projections
        self.query_rgb = nn.Linear(feature_dim, feature_dim)
        self.key_depth = nn.Linear(feature_dim, feature_dim)
        self.value_depth = nn.Linear(feature_dim, feature_dim)
        self.scale = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

        # Output projection
        self.out_proj = nn.Linear(feature_dim, feature_dim)

    def forward(self, rgb_features, depth_features):
        """
        Args:
            rgb_features: [batch_size, seq_len, feature_dim]
            depth_features: [batch_size, seq_len, feature_dim]
        
        Returns:
            Fused features: [batch_size, seq_len, feature_dim]
        """
        batch_size, seq_len, _ = rgb_features.size()

        # Compute Q, K, V
        Q = self.query_rgb(rgb_features)  # [batch_size, seq_len, feature_dim]
        K = self.key_depth(depth_features)  # [batch_size, seq_len, feature_dim]
        V = self.value_depth(depth_features)  # [batch_size, seq_len, feature_dim]

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [batch_size, num_heads, seq_len, seq_len]
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)  # [batch_size, num_heads, seq_len, head_dim]

        # Reshape and project
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.feature_dim)  # [batch_size, seq_len, feature_dim]
        fused = self.out_proj(context)

        return fused

# Direct Depth to Translation Estimator
class DepthTranslationEstimator(nn.Module):
    def __init__(self, depth_feature_dim=512, hidden_dim=256):
        super(DepthTranslationEstimator, self).__init__()
        self.estimator = nn.Sequential(
            nn.Linear(depth_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3)  # 3D translation (x, y, z)
        )
        
    def forward(self, depth_features):
        """
        Estimate translation directly from depth features
        
        Args:
            depth_features: Depth features [batch_size, seq_len, feature_dim]
            
        Returns:
            Estimated translation [batch_size, seq_len, 3]
        """
        return self.estimator(depth_features)

# Direct GPS to Translation Estimator
class GPSTranslationEstimator(nn.Module):
    def __init__(self, gps_feature_dim=512, hidden_dim=256):
        super(GPSTranslationEstimator, self).__init__()
        self.estimator = nn.Sequential(
            nn.Linear(gps_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3)  # 3D translation (x, y, z)
        )
        
    def forward(self, gps_features):
        """
        Estimate translation directly from GPS features
        
        Args:
            gps_features: GPS features [batch_size, seq_len, feature_dim]
            
        Returns:
            Estimated translation [batch_size, seq_len, 3]
        """
        return self.estimator(gps_features)

# GPS Feature Extractor
class GPSFeatureExtractor(nn.Module):
    def __init__(self, input_size=9, hidden_size=128, output_size=256, dropout=0.2):
        """
        Args:
            input_size: Number of GPS features (9: position [x, y, z], velocity [vx, vy, vz], signal [num_sats, GDOP, PDOP])
            hidden_size: Size of hidden layers
            output_size: Size of output features
            dropout: Dropout probability
        """
        super(GPSFeatureExtractor, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Feature extraction network
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
            nn.LeakyReLU(0.1)
        )
        
    def forward(self, gps_features):
        """
        Process GPS features.
        
        Args:
            gps_features: Tensor of shape [batch_size, seq_len, input_size]
        
        Returns:
            GPS features of shape [batch_size, seq_len, output_size]
        """
        batch_size, seq_len, _ = gps_features.shape
        
        # Reshape for batch processing
        x = gps_features.view(-1, self.input_size)
        
        # Extract features
        x = self.encoder(x)
        
        # Reshape back
        x = x.view(batch_size, seq_len, self.output_size)
        
        return x

class VisualInertialOdometryModel(nn.Module):
    """
    Visual-Inertial Odometry model that combines RGB images, Depth, IMU, and GPS data.
    Uses FlowNet for RGB, MonoDepth2 encoder for Depth, cross-modal attention for fusion,
    IMU for rotation improvement, and GPS for translation improvement.
    """
    def __init__(self, 
                 imsize1=184, 
                 imsize2=608, 
                 batchNorm=True,
                 use_imu=True,
                 imu_feature_size=128,
                 imu_input_size=21,
                 use_gps=True,
                 gps_feature_size=128,
                 gps_input_size=9,
                 use_adaptive_weighting=True,
                 use_depth_translation=True,
                 use_gps_translation=True,
                 pretrained_depth_path=None):
        super(VisualInertialOdometryModel, self).__init__()
        
        self.imsize1 = imsize1
        self.imsize2 = imsize2
        self.batchNorm = batchNorm
        self.use_imu = use_imu
        self.imu_feature_size = imu_feature_size
        self.imu_input_size = imu_input_size
        self.use_gps = use_gps
        self.gps_feature_size = gps_feature_size
        self.gps_input_size = gps_input_size
        self.use_depth_translation = use_depth_translation
        self.use_gps_translation = use_gps_translation
        
        # FlowNet-based encoder for RGB
        self.conv1   = conv(self.batchNorm,   6,   64, kernel_size=7, stride=2, dropout=par.conv_dropout[0])
        self.conv2   = conv(self.batchNorm,  64,  128, kernel_size=5, stride=2, dropout=par.conv_dropout[1])
        self.conv3   = conv(self.batchNorm, 128,  256, kernel_size=5, stride=2, dropout=par.conv_dropout[2])
        self.conv3_1 = conv(self.batchNorm, 256,  256, kernel_size=3, stride=1, dropout=par.conv_dropout[3])
        self.conv4   = conv(self.batchNorm, 256,  512, kernel_size=3, stride=2, dropout=par.conv_dropout[4])
        self.conv4_1 = conv(self.batchNorm, 512,  512, kernel_size=3, stride=1, dropout=par.conv_dropout[5])
        self.conv5   = conv(self.batchNorm, 512,  512, kernel_size=3, stride=2, dropout=par.conv_dropout[6])
        self.conv5_1 = conv(self.batchNorm, 512,  512, kernel_size=3, stride=1, dropout=par.conv_dropout[7])
        self.conv6   = conv(self.batchNorm, 512, 1024, kernel_size=3, stride=2, dropout=par.conv_dropout[8])
        
        # Calculate feature size after RGB encoding
        with torch.no_grad():
            dummy_input = torch.zeros(1, 6, imsize1, imsize2)
            dummy_output = self.encode_image(dummy_input)
            self.rgb_feature_size = dummy_output.numel()
        
        # Depth encoder using MonoDepth2's ResNet-18 encoder
        self.depth_encoder = MonoDepthEncoder(
            in_channels=2,  # Depth pairs (2 consecutive frames)
            num_features=512,  # Output feature dimension
            pretrained_path=pretrained_depth_path
        )
        
        # Project RGB and Depth features to a common space
        self.rgb_proj = nn.Linear(self.rgb_feature_size, 512)
        self.depth_proj = nn.Linear(512, 512)  # Depth encoder outputs 512 features
        
        # Cross-modal attention to fuse RGB and Depth
        self.cross_attention = CrossModalAttention(feature_dim=512, num_heads=8)
        
        # IMU processing network (if enabled)
        if self.use_imu:
            self.imu_feature_extractor = IMUFeatureExtractor(
                input_size=self.imu_input_size,
                hidden_size=self.imu_feature_size,
                output_size=self.imu_feature_size,
                dropout=0.2
            )
            
            # Fusion layer for visual (RGB+Depth) and IMU features
            self.fusion_layer = VisualInertialFusion(
                visual_size=512,  # After RGB-Depth fusion
                imu_size=self.imu_feature_size,
                fusion_size=par.rnn_hidden_size,
                use_adaptive_weighting=use_adaptive_weighting
            )
        
        # GPS processing network (if enabled)
        if self.use_gps:
            self.gps_feature_extractor = GPSFeatureExtractor(
                input_size=self.gps_input_size,
                hidden_size=self.gps_feature_size,
                output_size=self.gps_feature_size,
                dropout=0.2
            )
            
            # Fusion layer for visual+IMU and GPS features
            self.gps_fusion_layer = VisualInertialFusion(
                visual_size=par.rnn_hidden_size if self.use_imu else 512,
                imu_size=self.gps_feature_size,  # Treat GPS features like IMU for fusion
                fusion_size=par.rnn_hidden_size,
                use_adaptive_weighting=use_adaptive_weighting
            )
        
        # LSTM layer for sequential processing
        lstm_input_size = par.rnn_hidden_size if self.use_imu or self.use_gps else 512
        self.rnn = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=par.rnn_hidden_size,
            num_layers=2,
            dropout=par.rnn_dropout_between,
            batch_first=True
        )
        
        # Output head
        self.rnn_dropout = nn.Dropout(par.rnn_dropout_out)
        self.fc = nn.Linear(in_features=par.rnn_hidden_size, out_features=6)
        
        # Depth-to-translation estimator
        if self.use_depth_translation:
            self.depth_translation_estimator = DepthTranslationEstimator(
                depth_feature_dim=512,
                hidden_dim=256
            )
            
            # Fusion module to combine LSTM-based and depth-based translation
            self.translation_fusion = nn.Sequential(
                nn.Linear(6, 32),  # 3 from LSTM-based + 3 from depth-based
                nn.ReLU(),
                nn.Linear(32, 3)
            )
        
        # GPS-to-translation estimator
        if self.use_gps_translation:
            self.gps_translation_estimator = GPSTranslationEstimator(
                gps_feature_dim=self.gps_feature_size,
                hidden_dim=256
            )
            
            # Fusion module to combine LSTM-based, depth-based, and GPS-based translation
            input_size = 9 if self.use_depth_translation else 6  # 3 (LSTM) + 3 (depth) + 3 (GPS) or 3 (LSTM) + 3 (GPS)
            self.translation_fusion = nn.Sequential(
                nn.Linear(input_size, 32),
                nn.ReLU(),
                nn.Linear(32, 3)
            )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    constant_(m.bias.data, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        kaiming_normal_(param.data)
                    elif 'bias' in name:
                        constant_(param.data, 0)
                        # Set forget gate bias to 1
                        if 'bias_ih' in name or 'bias_hh' in name:
                            n = param.size(0)
                            param.data[n//4:n//2].fill_(1.)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def encode_image(self, x):
        """Encode RGB image pairs using FlowNet architecture."""
        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6(out_conv5)
        return out_conv6
    
    def forward(self, rgb, depth=None, imu_data=None, gps_data=None):
        """
        Forward pass through the network.
        
        Args:
            rgb: Input tensor of shape [batch_size, seq_len+1, 3, H, W]
            depth: Depth tensor of shape [batch_size, seq_len+1, 1, H, W]
            imu_data: Optional IMU tensor of shape [batch_size, seq_len, imu_input_size]
            gps_data: Optional GPS tensor of shape [batch_size, seq_len, gps_input_size]
        
        Returns:
            Tensor of shape [batch_size, seq_len, 6] with predicted relative poses
            (roll, pitch, yaw, x, y, z)
        """
        batch_size = rgb.size(0)
        seq_len = rgb.size(1) - 1  # Number of frame pairs
        
        # Process RGB pairs
        rgb_pairs = []
        for i in range(seq_len):
            pair = torch.cat([rgb[:, i], rgb[:, i+1]], dim=1)  # [batch_size, 6, H, W]
            rgb_pairs.append(pair)
        
        rgb = torch.stack(rgb_pairs, dim=1)  # [batch_size, seq_len, 6, H, W]
        rgb = rgb.view(batch_size * seq_len, 6, self.imsize1, self.imsize2)
        
        # Encode RGB features
        rgb_features = self.encode_image(rgb)  # [batch_size * seq_len, 1024, H', W']
        rgb_features = rgb_features.view(batch_size, seq_len, -1)  # [batch_size, seq_len, rgb_feature_size]
        
        # Project RGB features
        rgb_features = self.rgb_proj(rgb_features)  # [batch_size, seq_len, 512]
        
        # Initialize depth and GPS translation estimations
        depth_translation = None
        gps_translation = None
        
        # Process Depth pairs (if provided)
        if depth is not None:
            depth_pairs = []
            for i in range(seq_len):
                pair = torch.cat([depth[:, i], depth[:, i+1]], dim=1)  # [batch_size, 2, H, W]
                depth_pairs.append(pair)
            
            depth = torch.stack(depth_pairs, dim=1)  # [batch_size, seq_len, 2, H, W]
            depth = depth.view(batch_size * seq_len, 2, self.imsize1, self.imsize2)
            
            # Encode Depth features
            depth_features = self.depth_encoder(depth)  # [batch_size * seq_len, 512]
            depth_features = depth_features.view(batch_size, seq_len, -1)  # [batch_size, seq_len, 512]
            
            # Project Depth features
            depth_features = self.depth_proj(depth_features)  # [batch_size, seq_len, 512]
            
            # Direct translation estimation from depth
            if self.use_depth_translation:
                depth_translation = self.depth_translation_estimator(depth_features)
            
            # Fuse RGB and Depth using cross-modal attention
            fused_features = self.cross_attention(rgb_features, depth_features)  # [batch_size, seq_len, 512]
        else:
            fused_features = rgb_features  # Use RGB features alone if depth is not provided
        
        # Process IMU data if available
        if self.use_imu and imu_data is not None:
            # Process IMU data with feature extractor
            imu_features = self.imu_feature_extractor(imu_data)
            
            # Fuse visual (RGB+Depth) and IMU features
            fused_features = self.fusion_layer(fused_features, imu_features)
        
        # Process GPS data if available
        if self.use_gps and gps_data is not None:
            # Process GPS data with feature extractor
            gps_features = self.gps_feature_extractor(gps_data)
            
            # Direct translation estimation from GPS
            if self.use_gps_translation:
                gps_translation = self.gps_translation_estimator(gps_features)
            
            # Fuse visual+IMU and GPS features
            fused_features = self.gps_fusion_layer(fused_features, gps_features)
        
        # Process with LSTM
        x, _ = self.rnn(fused_features)
        x = self.rnn_dropout(x)
        
        # Predict relative poses
        pred = self.fc(x)  # [batch_size, seq_len, 6]
        
        # Apply depth and GPS-based translation improvements if available
        if (self.use_depth_translation and depth_translation is not None) or \
           (self.use_gps_translation and gps_translation is not None):
            # Split into rotation and translation components
            pred_rot = pred[:, :, :3]  # Roll, pitch, yaw
            pred_trans = pred[:, :, 3:]  # x, y, z
            
            # Combine LSTM-based, depth-based, and GPS-based translation
            trans_inputs = [pred_trans]
            if self.use_depth_translation and depth_translation is not None:
                trans_inputs.append(depth_translation)
            if self.use_gps_translation and gps_translation is not None:
                trans_inputs.append(gps_translation)
            
            combined_trans_input = torch.cat(trans_inputs, dim=2)
            refined_trans = self.translation_fusion(combined_trans_input)
            
            # Recombine with rotation
            pred = torch.cat([pred_rot, refined_trans], dim=2)
            
            # Store depth and GPS translations for loss computation
            if self.use_depth_translation:
                self.last_depth_translation = depth_translation
            else:
                self.last_depth_translation = None
            if self.use_gps_translation:
                self.last_gps_translation = gps_translation
            else:
                self.last_gps_translation = None
        else:
            self.last_depth_translation = None
            self.last_gps_translation = None
        
        return pred
    
    def get_loss(self, pred, target):
        """
        Compute pose prediction loss with depth and GPS-supervised translation.
        
        Args:
            pred: Predicted relative poses [batch_size, seq_len, 6]
            target: Target relative poses [batch_size, seq_len, 6]
            
        Returns:
            Tuple of (total_loss, rot_loss, trans_loss, depth_trans_loss, gps_trans_loss)
        """
        # Split into rotation and translation components
        pred_rot = pred[:, :, :3]
        pred_trans = pred[:, :, 3:]
        target_rot = target[:, :, :3]
        target_trans = target[:, :, 3:]
        
        # Compute rotation and translation losses
        rot_loss = F.mse_loss(pred_rot, target_rot)
        trans_loss = F.mse_loss(pred_trans, target_trans)
        
        # Compute depth-translation loss if available
        depth_trans_loss = torch.tensor(0.0, device=pred.device)
        if self.use_depth_translation and hasattr(self, 'last_depth_translation') and self.last_depth_translation is not None:
            depth_trans_loss = F.mse_loss(self.last_depth_translation, target_trans)
        
        # Compute GPS-translation loss if available
        gps_trans_loss = torch.tensor(0.0, device=pred.device)
        if self.use_gps_translation and hasattr(self, 'last_gps_translation') and self.last_gps_translation is not None:
            gps_trans_loss = F.mse_loss(self.last_gps_translation, target_trans)
        
        # L2 regularization
        l2_lambda = 0.001
        l2_reg = sum(param.pow(2).sum() for param in self.parameters())
    
        # Total loss with weighted components
        #total_loss = rot_loss + 50 * trans_loss + depth_trans_loss + gps_trans_loss + l2_lambda * l2_reg
        total_loss = 100 * rot_loss + 10 * trans_loss + gps_trans_loss + l2_lambda * l2_reg
        
        return total_loss, rot_loss, trans_loss, depth_trans_loss, gps_trans_loss

def load_pretrained_flownet(model, flownet_path):
    """
    Load pretrained FlowNet weights into the model.
    
    Args:
        model: VisualInertialOdometryModel model
        flownet_path: Path to FlowNet weights
        
    Returns:
        bool: Success status
    """
    if not os.path.exists(flownet_path):
        print(f"FlowNet weights not found at: {flownet_path}")
        return False
    
    try:
        print(f"Loading pretrained FlowNet weights from: {flownet_path}")
        checkpoint = torch.load(flownet_path, map_location='cpu')
        
        # Check if weights are stored under 'state_dict'
        if 'state_dict' in checkpoint:
            pretrained_weights = checkpoint['state_dict']
        else:
            pretrained_weights = checkpoint
        
        # Print the keys to see what's available
        print("Available keys in pretrained weights:")
        key_sample = list(pretrained_weights.keys())[:10]
        for key in key_sample:
            print(f"  - {key}")
        
        # Map FlowNet layers to our model
        conv_layers_map = {
            'conv1.0': model.conv1[0],
            'conv2.0': model.conv2[0],
            'conv3.0': model.conv3[0],
            'conv3_1.0': model.conv3_1[0],
            'conv4.0': model.conv4[0],
            'conv4_1.0': model.conv4_1[0],
            'conv5.0': model.conv5[0],
            'conv5_1.0': model.conv5_1[0],
            'conv6.0': model.conv6[0]
        }
        
        # Load convolutional weights
        matched_layers = 0
        for layer_name, layer in conv_layers_map.items():
            weight_key = f"{layer_name}.weight"
            if weight_key in pretrained_weights:
                if layer.weight.data.shape == pretrained_weights[weight_key].shape:
                    layer.weight.data.copy_(pretrained_weights[weight_key])
                    print(f"Loaded weights for: {layer_name}")
                    matched_layers += 1
                else:
                    print(f"Shape mismatch for {layer_name}: {layer.weight.data.shape} vs {pretrained_weights[weight_key].shape}")
            else:
                print(f"Key not found: {weight_key}")
        
        # Also load batch norm weights if applicable
        if model.batchNorm:
            bn_layers_map = {
                'conv1.1': model.conv1[1],
                'conv2.1': model.conv2[1],
                'conv3.1': model.conv3[1],
                'conv3_1.1': model.conv3_1[1],
                'conv4.1': model.conv4[1],
                'conv4_1.1': model.conv4_1[1],
                'conv5.1': model.conv5[1],
                'conv5_1.1': model.conv5_1[1],
                'conv6.1': model.conv6[1] if len(model.conv6) > 1 else None
            }
            
            for layer_name, layer in bn_layers_map.items():
                if layer is None:
                    continue
                    
                weight_key = f"{layer_name}.weight"
                bias_key = f"{layer_name}.bias"
                mean_key = f"{layer_name}.running_mean"
                var_key = f"{layer_name}.running_var"
                
                if all(k in pretrained_weights for k in [weight_key, bias_key, mean_key, var_key]):
                    if (layer.weight.data.shape == pretrained_weights[weight_key].shape and
                        layer.bias.data.shape == pretrained_weights[bias_key].shape and
                        layer.running_mean.shape == pretrained_weights[mean_key].shape and
                        layer.running_var.shape == pretrained_weights[var_key].shape):
                        
                        layer.weight.data.copy_(pretrained_weights[weight_key])
                        layer.bias.data.copy_(pretrained_weights[bias_key])
                        layer.running_mean.copy_(pretrained_weights[mean_key])
                        layer.running_var.copy_(pretrained_weights[var_key])
                        print(f"Loaded BatchNorm for: {layer_name}")
                        matched_layers += 1
                    else:
                        print(f"Shape mismatch for BatchNorm {layer_name}")
                else:
                    print(f"BatchNorm keys not found for: {layer_name}")
        
        if matched_layers > 0:
            print(f"Successfully loaded weights for {matched_layers} layers")
            return True
        else:
            print("Failed to match any layers")
            return False
            
    except Exception as e:
        print(f"Error loading FlowNet weights: {e}")
        import traceback
        traceback.print_exc()
        return False