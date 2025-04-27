# In params.py:

import os
import torch
import numpy as np

class Parameters:
    def __init__(self):
        # Directories
        self.data_dir = '/media/krkavinda/New Volume/Mid-Air-Dataset/MidAir_processed'
        self.image_dir = self.data_dir
        self.pose_dir = self.data_dir
        
        # Model parameters
        self.model_type = 'rgb'
        self.img_h = 184
        self.img_w = 608
        self.seq_len = 7
        self.batch_size = 24
        self.minus_point_5 = True
        
        # RGB stats
        self.img_means_rgb = [0.485, 0.456, 0.406]
        self.img_stds_rgb = [0.229, 0.224, 0.225]

        # Depth stats
        self.depth_mean = 23.0  # Placeholder; update with computed value
        self.depth_std = 5.0   # Placeholder; update with computed value

        # Add flags for modalities
        self.use_imu = True
        self.use_gps = True   # New flag for GPS
        self.imu_feature_size = 512

        self.use_depth = True
        self.use_depth_temporal = True
        self.use_depth_translation =  True


        
        # NEW FLAG - Use camera frame data
        self.use_camera_frame = True
        
        # Coordinate transformation settings
        if self.use_camera_frame:
            # Use identity matrix when using camera frame data (already transformed)
            self.body_to_camera = torch.tensor([
                [1, 0, 0], 
                [0, 1, 0], 
                [0, 0, 1]], dtype=torch.float32)
        else:
            # Use the correct transformation matrix when using body frame data
            self.body_to_camera = torch.tensor([
                [0, 1, 0], 
                [0, 0, -1], 
                [-1, 0, 0]], dtype=torch.float32)
        
        # Translation offset from Body to Camera frame (in meters, NED)
        self.body_to_camera_translation = torch.tensor([1.0, 0.0, 0.5], dtype=torch.float32)
        
        # Network parameters
        self.rnn_hidden_size = 1000
        self.rnn_dropout_out = 0.5
        self.rnn_dropout_between = 0.0
        self.conv_dropout = (0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0.3, 0.3, 0.7)
        self.batch_norm = True
        self.rot_weight = 100
        
        # Training parameters
        self.epochs = 100
        self.n_processors = 25
        self.pin_mem = True
        self.optim = {'opt': 'Adam', 'lr': 5e-4, 'weight_decay': 0.0005}
        
        # FlowNet weights path
        self.pretrained_flownet = '/home/krkavinda/PROJECT-IGI3/FlowNet_models/pytorch/flownets_bn_EPE2.459.pth'
        self.pretrained_depth = '/home/krkavinda/PROJECT-IGI3/mono_640x192/encoder.pth'
        
        # Trajectories for training and validation
        self.train_trajectories = [
            ('Kite_training/sunny', 'trajectory_0001'),
            ('Kite_training/sunny', 'trajectory_0003'),
            # ('Kite_training/sunny', 'trajectory_0007'),
            # ('Kite_training/sunny', 'trajectory_0011'),
            ('Kite_training/sunny', 'trajectory_0013'),
            # ('Kite_training/sunny', 'trajectory_0015'),
            # ('Kite_training/sunny', 'trajectory_0017'),
            ('Kite_training/sunny', 'trajectory_0019'),
            # ('Kite_training/sunny', 'trajectory_0023'),
            # ('Kite_training/sunny', 'trajectory_0025'),
            ('Kite_training/sunny', 'trajectory_0027'),
            ('Kite_training/sunny', 'trajectory_0029'),
        ]
        
        self.valid_trajectories = [
            ('Kite_training/sunny', f'trajectory_{i:04d}')
            for i in range(0, 15, 4)
        ]

        self.test_trajectories = [
            ('Kite_training/sunny', 'trajectory_0008'),
        ]
        
        # Output paths
        self.model_dir = './models'
        self.log_dir = './logs'
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.model_name = f"rgbvo_im{self.img_h}x{self.img_w}_s{self.seq_len}_b{self.batch_size}"
        self.model_path = '/home/krkavinda/PROJECT-IGI3/models/rgbvo_im184x608_s7_b24_with_imu_integrated_with_gps_with_depth_best_ate.pt'

# Create global parameters instance
par = Parameters()