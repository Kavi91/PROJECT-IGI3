import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import math
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

# Depth Temporal Branch
class DepthTemporalBranch(nn.Module):
    def __init__(self, input_size=512, hidden_size=512, num_layers=1, dropout=0.2):
        super(DepthTemporalBranch, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: Depth features [batch_size, seq_len, input_size]
        Returns:
            Temporal depth features [batch_size, seq_len, hidden_size]
        """
        output, _ = self.lstm(x)
        output = self.norm(output)
        output = self.dropout(output)
        return output

# GPS Temporal Branch
class GPSTemporalBranch(nn.Module):
    def __init__(self, input_size=512, hidden_size=512, num_layers=1, dropout=0.2):
        super(GPSTemporalBranch, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: GPS features [batch_size, seq_len, input_size]
        Returns:
            Temporal GPS features [batch_size, seq_len, hidden_size]
        """
        output, _ = self.lstm(x)
        output = self.norm(output)
        output = self.dropout(output)
        return output

# Multi-Modal Self-Attention Fusion
class MultiModalSelfAttention(nn.Module):
    def __init__(self, feature_dim=512, num_heads=8, dropout=0.1):
        super(MultiModalSelfAttention, self).__init__()
        
        # Ensure feature_dim is divisible by num_heads
        assert feature_dim % num_heads == 0, "Feature dimension must be divisible by number of heads"
        
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        
        # Projections for the three modalities
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        
        # Output projection
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        
        # Layer norm before and after
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        
        # Feed-forward network after attention
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 4, feature_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, main_features, depth_features, gps_features):
        """
        Fuse different modality features using self-attention
        
        Args:
            main_features: Main branch features [batch_size, seq_len, feature_dim]
            depth_features: Depth branch features [batch_size, seq_len, feature_dim]
            gps_features: GPS branch features [batch_size, seq_len, feature_dim]
            
        Returns:
            Fused features [batch_size, seq_len, feature_dim]
        """
        batch_size, seq_len, _ = main_features.shape
        
        # Concatenate all modalities along sequence dimension to treat as a longer sequence
        # Shape: [batch_size, 3*seq_len, feature_dim]
        concat_features = torch.cat([main_features, depth_features, gps_features], dim=1)
        concat_features = self.norm1(concat_features)
        
        # Multi-head self-attention
        q = self.q_proj(concat_features)
        k = self.k_proj(concat_features)
        v = self.v_proj(concat_features)
        
        # Reshape for multi-head attention
        head_dim = self.feature_dim // self.num_heads
        q = q.view(batch_size, 3*seq_len, self.num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, 3*seq_len, self.num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, 3*seq_len, self.num_heads, head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back to original shape
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, 3*seq_len, self.feature_dim)
        attn_output = self.out_proj(attn_output)
        
        # Add residual connection
        concat_features = concat_features + self.dropout(attn_output)
        
        # Extract features for each modality - get fused representations of each
        main_fused = concat_features[:, :seq_len]
        depth_fused = concat_features[:, seq_len:2*seq_len]
        gps_fused = concat_features[:, 2*seq_len:]
        
        # Final feature is the main branch with feed-forward network
        output = self.norm2(main_fused)
        output = main_fused + self.dropout(self.ffn(output))
        
        return output, depth_fused, gps_fused

# Hybrid Temporal Module combining self-attention and LSTM
class HybridTemporalModule(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads=8, num_layers=2, dropout=0.1):
        super(HybridTemporalModule, self).__init__()
        
        # Self-attention for temporal relationships
        self.norm1 = nn.LayerNorm(input_size)
        self.self_attn = nn.MultiheadAttention(input_size, num_heads, dropout=dropout, batch_first=True)
        
        # Feed-forward layer after attention
        self.norm2 = nn.LayerNorm(input_size)
        self.ffn = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, input_size)
        )
        
        # LSTM for sequential processing
        self.norm3 = nn.LayerNorm(input_size)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output projection (if hidden_size != input_size)
        self.proj = nn.Linear(hidden_size, input_size)
        self.norm4 = nn.LayerNorm(input_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, input_size]
        
        Returns:
            Output tensor with shape [batch_size, seq_len, input_size]
        """
        # Layer normalization before self-attention
        residual = x
        x = self.norm1(x)
        
        # Multi-head self-attention
        attn_output, _ = self.self_attn(x, x, x)
        x = residual + self.dropout(attn_output)
        
        # Feed-forward network
        residual = x
        x = self.norm2(x)
        x = residual + self.dropout(self.ffn(x))
        
        # LSTM layer
        residual = x
        x = self.norm3(x)
        lstm_out, _ = self.lstm(x)
        
        # Project LSTM output back to input size if needed
        lstm_out = self.proj(lstm_out)
        
        # Add residual and apply layer norm
        x = residual + self.dropout(lstm_out)
        x = self.norm4(x)
        
        return x

# GPS Feature Extractor
class GPSFeatureExtractor(nn.Module):
    def __init__(self, input_size=9, hidden_size=128, output_size=512, dropout=0.2):
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
                 imu_feature_size=512,
                 imu_input_size=21,
                 use_gps=True,
                 gps_feature_size=512,
                 gps_input_size=9,
                 use_adaptive_weighting=True,
                 use_depth=False,
                 use_depth_temporal=None,
                 use_depth_translation=None,
                 use_gps_temporal=True,
                 pretrained_depth_path=None):
        super(VisualInertialOdometryModel, self).__init__()
        
        # Set default values for depth-related parameters if not provided
        if use_depth_temporal is None:
            use_depth_temporal = use_depth
        if use_depth_translation is None:
            use_depth_translation = use_depth
        
        self.imsize1 = imsize1
        self.imsize2 = imsize2
        self.batchNorm = batchNorm
        self.use_imu = use_imu
        self.imu_feature_size = imu_feature_size
        self.imu_input_size = imu_input_size
        self.use_gps = use_gps
        self.gps_feature_size = gps_feature_size
        self.gps_input_size = gps_input_size
        self.use_gps_temporal = use_gps and use_gps_temporal

        self.use_depth = use_depth
        self.use_depth_temporal = use_depth and use_depth_temporal
        self.use_depth_translation = use_depth and use_depth_translation
        
        print(f"Model configuration - IMU: {use_imu}, GPS: {use_gps}, Depth: {use_depth}")
        print(f"Depth flags - use_depth: {self.use_depth}, use_depth_temporal: {self.use_depth_temporal}, use_depth_translation: {self.use_depth_translation}")
        
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
        
        # Depth encoder using MonoDepth2's ResNet-18 encoder (only instantiated if use_depth is True)
        if self.use_depth:
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
        else:
            # If not using depth, we still need the RGB projection for consistency
            self.rgb_proj = nn.Linear(self.rgb_feature_size, 512)
        
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
                fusion_size=512,
                use_adaptive_weighting=use_adaptive_weighting
            )
        
        # GPS processing network (if enabled)
        if self.use_gps:
            self.gps_feature_extractor = GPSFeatureExtractor(
                input_size=self.gps_input_size,
                hidden_size=256,
                output_size=self.gps_feature_size,
                dropout=0.2
            )
            
            # Fusion layer for visual+IMU and GPS features
            self.gps_fusion_layer = VisualInertialFusion(
                visual_size=512 if self.use_imu else 512,
                imu_size=self.gps_feature_size,  # Treat GPS features like IMU for fusion
                fusion_size=512,
                use_adaptive_weighting=use_adaptive_weighting
            )
        
        # New temporal branches for depth and GPS
        if self.use_depth_temporal:
            self.depth_temporal_branch = DepthTemporalBranch(
                input_size=512,  # Depth features size
                hidden_size=512,
                num_layers=1,
                dropout=0.2
            )
            
            # Depth pose prediction layer
            self.depth_pose_predictor = nn.Linear(in_features=512, out_features=6)
        
        if self.use_gps_temporal and self.use_gps:
            self.gps_temporal_branch = GPSTemporalBranch(
                input_size=self.gps_feature_size,
                hidden_size=512,
                num_layers=1, 
                dropout=0.2
            )
            
            # GPS pose prediction layer
            self.gps_pose_predictor = nn.Linear(in_features=512, out_features=6)
            
        # Main branch temporal module
        self.temporal_module = HybridTemporalModule(
            input_size=512, 
            hidden_size=par.rnn_hidden_size,
            num_heads=8,
            num_layers=2,
            dropout=par.rnn_dropout_between
        )
        
        # Multi-modal self-attention fusion for all branches
        self.multi_modal_fusion = MultiModalSelfAttention(
            feature_dim=512,
            num_heads=8,
            dropout=0.1
        )
        
        # Main pose predictor
        self.main_pose_predictor = nn.Linear(in_features=512, out_features=6)
        
        # Final pose fusion layer
        pose_input_size = 6
        if self.use_depth_temporal:
            pose_input_size += 6
        if self.use_gps_temporal and self.use_gps:
            pose_input_size += 6
            
        self.pose_fusion = nn.Sequential(
            nn.Linear(pose_input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 6)
        )
        
        # Dropout for regularization
        self.rnn_dropout = nn.Dropout(par.rnn_dropout_out)
        
        # Initialize weights
        self._init_weights()
        
        # Initialize variables for loss computation
        self.last_main_pose = None
        self.last_depth_pose = None
        self.last_gps_pose = None
    
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
            depth: Optional depth tensor of shape [batch_size, seq_len+1, 1, H, W]
            imu_data: Optional IMU tensor of shape [batch_size, seq_len, imu_input_size]
            gps_data: Optional GPS tensor of shape [batch_size, seq_len, gps_input_size]
        
        Returns:
            Tensor of shape [batch_size, seq_len, 6] with predicted relative poses
            (roll, pitch, yaw, x, y, z)
        """
        # Initialize variables for multi-branch pose prediction
        depth_pose = None
        gps_pose = None
        
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
        
        # Process Depth pairs (if provided and depth is enabled)
        depth_features = None
        if depth is not None and self.use_depth:
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
            
            # Fuse RGB and Depth using cross-modal attention
            fused_features = self.cross_attention(rgb_features, depth_features)  # [batch_size, seq_len, 512]
        else:
            fused_features = rgb_features  # Use RGB features alone if depth is not provided
            depth_features = torch.zeros_like(rgb_features)  # Create dummy depth features
        
        # Process IMU data if available and IMU is enabled
        if self.use_imu and imu_data is not None:
            # Process IMU data with feature extractor
            imu_features = self.imu_feature_extractor(imu_data)
            
            # Fuse visual (RGB+Depth) and IMU features
            fused_features = self.fusion_layer(fused_features, imu_features)
        
        # Process GPS data if available and GPS is enabled
        gps_features = None
        if self.use_gps and gps_data is not None:
            # Process GPS data with feature extractor
            gps_features = self.gps_feature_extractor(gps_data)
            
            # Fuse visual+IMU and GPS features
            fused_features = self.gps_fusion_layer(fused_features, gps_features)
        else:
            gps_features = torch.zeros_like(fused_features)  # Create dummy GPS features
        
        # Process through temporal branches
        main_temporal_features = self.temporal_module(fused_features)
        
        # Process depth branch if enabled
        depth_temporal_features = None
        if self.use_depth_temporal and depth is not None:
            depth_temporal_features = self.depth_temporal_branch(depth_features)
        else:
            depth_temporal_features = torch.zeros_like(main_temporal_features)
        
        # Process GPS branch if enabled
        gps_temporal_features = None
        if self.use_gps_temporal and self.use_gps and gps_data is not None:
            gps_temporal_features = self.gps_temporal_branch(gps_features)
        else:
            gps_temporal_features = torch.zeros_like(main_temporal_features)
        
        # Apply multi-modal self-attention fusion
        fused_main, fused_depth, fused_gps = self.multi_modal_fusion(
            main_temporal_features, 
            depth_temporal_features, 
            gps_temporal_features
        )
        
        # Predict poses from each branch
        main_pose = self.main_pose_predictor(self.rnn_dropout(fused_main))
        
        # Collect all pose predictions
        pose_inputs = [main_pose]
        
        if self.use_depth_temporal and depth is not None:
            depth_pose = self.depth_pose_predictor(fused_depth)
            pose_inputs.append(depth_pose)
        
        if self.use_gps_temporal and self.use_gps and gps_data is not None:
            gps_pose = self.gps_pose_predictor(fused_gps)
            pose_inputs.append(gps_pose)
        
        # Fuse pose predictions if using multiple branches
        if len(pose_inputs) > 1:
            combined_pose = torch.cat(pose_inputs, dim=2)
            final_pose = self.pose_fusion(combined_pose)
        else:
            final_pose = main_pose
        
        # Store individual branch results for loss computation
        self.last_main_pose = main_pose
        self.last_depth_pose = depth_pose  # Will be None if not computed
        self.last_gps_pose = gps_pose      # Will be None if not computed
        
        return final_pose
    
    def get_loss(self, pred, target):
        """
        Compute pose prediction loss with branch-specific supervision.
        
        Args:
            pred: Predicted relative poses [batch_size, seq_len, 6]
            target: Target relative poses [batch_size, seq_len, 6]
            
        Returns:
            Tuple of (total_loss, rot_loss, trans_loss, depth_loss, gps_loss)
        """
        # Main prediction loss
        rot_loss = F.mse_loss(pred[:, :, :3], target[:, :, :3])
        trans_loss = F.mse_loss(pred[:, :, 3:], target[:, :, 3:])
        
        # Branch-specific losses
        depth_loss = torch.tensor(0.0, device=pred.device)
        
        if self.use_depth_temporal and self.last_depth_pose is not None:
            depth_rot_loss = F.mse_loss(self.last_depth_pose[:, :, :3], target[:, :, :3])
            depth_trans_loss = F.mse_loss(self.last_depth_pose[:, :, 3:], target[:, :, 3:])
            depth_loss = depth_rot_loss + depth_trans_loss
            #print(f"Depth loss: {depth_loss.item():.6f} (rot: {depth_rot_loss.item():.6f}, trans: {depth_trans_loss.item():.6f})")
        
        gps_loss = torch.tensor(0.0, device=pred.device)
        if self.use_gps_temporal and self.use_gps and self.last_gps_pose is not None:
            gps_rot_loss = F.mse_loss(self.last_gps_pose[:, :, :3], target[:, :, :3])
            gps_trans_loss = F.mse_loss(self.last_gps_pose[:, :, 3:], target[:, :, 3:])
            gps_loss = gps_rot_loss + gps_trans_loss
        
        # L2 regularization
        l2_lambda = 0.001
        l2_reg = sum(param.pow(2).sum() for param in self.parameters())
        
        # Weight the losses
        # You might want to tune these weights during training
        total_loss = 10*rot_loss + 5*trans_loss +  2*gps_loss + l2_lambda * l2_reg  #+ 5 * depth_loss 
        
        return total_loss, rot_loss, trans_loss, depth_loss, gps_loss

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