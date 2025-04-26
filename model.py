import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.init import kaiming_normal_, constant_

from params import par

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

class RGBVO(nn.Module):
    """RGB Visual Odometry model based on FlowNet + LSTM architecture."""
    
    def __init__(self, imsize1=184, imsize2=608, batchNorm=True):
        super(RGBVO, self).__init__()
        
        self.imsize1 = imsize1
        self.imsize2 = imsize2
        self.batchNorm = batchNorm
        
        # FlowNet-based encoder
        self.conv1   = conv(self.batchNorm,   6,   64, kernel_size=7, stride=2, dropout=par.conv_dropout[0])
        self.conv2   = conv(self.batchNorm,  64,  128, kernel_size=5, stride=2, dropout=par.conv_dropout[1])
        self.conv3   = conv(self.batchNorm, 128,  256, kernel_size=5, stride=2, dropout=par.conv_dropout[2])
        self.conv3_1 = conv(self.batchNorm, 256,  256, kernel_size=3, stride=1, dropout=par.conv_dropout[3])
        self.conv4   = conv(self.batchNorm, 256,  512, kernel_size=3, stride=2, dropout=par.conv_dropout[4])
        self.conv4_1 = conv(self.batchNorm, 512,  512, kernel_size=3, stride=1, dropout=par.conv_dropout[5])
        self.conv5   = conv(self.batchNorm, 512,  512, kernel_size=3, stride=2, dropout=par.conv_dropout[6])
        self.conv5_1 = conv(self.batchNorm, 512,  512, kernel_size=3, stride=1, dropout=par.conv_dropout[7])
        self.conv6   = conv(self.batchNorm, 512, 1024, kernel_size=3, stride=2, dropout=par.conv_dropout[8])
        
        # Calculate feature size after encoding
        with torch.no_grad():
            dummy_input = torch.zeros(1, 6, imsize1, imsize2)
            dummy_output = self.encode_image(dummy_input)
            self.feature_size = dummy_output.numel()
        
        # LSTM layer for sequential processing
        self.rnn = nn.LSTM(
            input_size=self.feature_size,
            hidden_size=par.rnn_hidden_size,
            num_layers=2,
            dropout=par.rnn_dropout_between,
            batch_first=True
        )
        
        # Output head
        self.rnn_dropout = nn.Dropout(par.rnn_dropout_out)
        self.fc = nn.Linear(in_features=par.rnn_hidden_size, out_features=6)
        
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
        """Encode image pairs using FlowNet architecture."""
        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6(out_conv5)
        return out_conv6
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len+1, 3, H, W]
               Contains seq_len+1 consecutive RGB frames
        
        Returns:
            Tensor of shape [batch_size, seq_len, 6] with predicted relative poses
            (roll, pitch, yaw, x, y, z)
        """
        batch_size = x.size(0)
        seq_len = x.size(1) - 1  # Number of frame pairs
        
        # Prepare frame pairs (each pair is concatenated along channel dimension)
        pairs = []
        for i in range(seq_len):
            pair = torch.cat([x[:, i], x[:, i+1]], dim=1)  # [batch_size, 6, H, W]
            pairs.append(pair)
        
        x = torch.stack(pairs, dim=1)  # [batch_size, seq_len, 6, H, W]
        x = x.view(batch_size * seq_len, 6, self.imsize1, self.imsize2)
        
        # Encode frame pairs
        x = self.encode_image(x)
        x = x.view(batch_size, seq_len, -1)
        
        # Process with LSTM
        x, _ = self.rnn(x)
        x = self.rnn_dropout(x)
        
        # Predict relative poses
        x = self.fc(x)
        
        return x
    
    def get_loss(self, pred, target):
        """
        Compute pose prediction loss.
        
        Args:
            pred: Predicted relative poses [batch_size, seq_len, 6]
            target: Target relative poses [batch_size, seq_len, 6]
            
        Returns:
            Total loss (weighted sum of rotation and translation losses)
        """
        # Split into rotation and translation components
        pred_rot = pred[:, :, :3]
        pred_trans = pred[:, :, 3:]
        target_rot = target[:, :, :3]
        target_trans = target[:, :, 3:]
        
        # Compute rotation and translation losses
        rot_loss = F.mse_loss(pred_rot, target_rot)
        trans_loss = F.mse_loss(pred_trans, target_trans)
        
        # Weighted total loss
        total_loss = par.rot_weight * rot_loss + trans_loss
        
        return total_loss, rot_loss, trans_loss

def load_pretrained_flownet(model, flownet_path):
    """
    Load pretrained FlowNet weights into the model.
    
    Args:
        model: RGBVO model
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
            'conv1': model.conv1[0],
            'conv2': model.conv2[0],
            'conv3': model.conv3[0],
            'conv3_1': model.conv3_1[0],
            'conv4': model.conv4[0],
            'conv4_1': model.conv4_1[0],
            'conv5': model.conv5[0],
            'conv5_1': model.conv5_1[0],
            'conv6': model.conv6[0]
        }
        
        # These are possible prefixes in FlowNet weights
        prefixes = ['', 'module.', 'net.']
        
        # Try to match and load weights for conv layers
        matched_layers = 0
        for layer_name, layer in conv_layers_map.items():
            matched = False
            for prefix in prefixes:
                key = f"{prefix}{layer_name}.weight"
                if key in pretrained_weights:
                    if layer.weight.data.shape == pretrained_weights[key].shape:
                        layer.weight.data.copy_(pretrained_weights[key])
                        matched = True
                        matched_layers += 1
                        break
            
            if not matched:
                print(f"No match found for layer: {layer_name}")
        
        if matched_layers > 0:
            print(f"Successfully loaded weights for {matched_layers} layers")
            return True
        else:
            print("Failed to match any layers")
            return False
            
    except Exception as e:
        print(f"Error loading FlowNet weights: {e}")
        return False

import os

if __name__ == "__main__":
    # Simple test
    model = RGBVO()
    print(f"Model created with feature size: {model.feature_size}")
    
    # Test with dummy input
    dummy_input = torch.randn(2, 6, 184, 608)  # [batch_size, seq_len+1, 3, H, W]
    dummy_output = model(dummy_input)
    print(f"Output shape: {dummy_output.shape}")