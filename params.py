import os
import torch
import numpy as np

class Parameters:
    def __init__(self):
        # Directories
        self.data_dir = '/media/krkavinda/New Volume/Mid-Air-Dataset/MidAir_processed'  # Change this to your data path
        self.image_dir = self.data_dir
        self.pose_dir = self.data_dir
        
        # Model parameters
        self.model_type = 'rgb'
        self.img_h = 184
        self.img_w = 608
        self.seq_len = 5  # Fixed sequence length
        self.batch_size = 24
        self.minus_point_5 = True  # FlowNet required normalization
        
        # RGB stats (can be updated with dataset-specific values)
        self.img_means_rgb = [0.485, 0.456, 0.406]
        self.img_stds_rgb = [0.229, 0.224, 0.225]

        # Add to your params.py if needed
        self.use_imu = True  # Whether to use IMU data
        self.imu_feature_size = 512  # Size of IMU features
        
        # Network parameters
        self.rnn_hidden_size = 1000
        self.rnn_dropout_out = 0.5
        self.rnn_dropout_between = 0.0
        self.conv_dropout = (0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0.3, 0.3, 0.7)
        self.batch_norm = True
        self.rot_weight = 100  # Weight for rotation loss
        
        # Training parameters
        self.epochs = 100
        self.n_processors = 25  # Adjust based on your system
        self.pin_mem = True
        self.optim = {'opt': 'Adam', 'lr': 1e-5, 'weight_decay': 0.0005}
        
        # Coordinate transformation matrix (identity by default)
        self.body_to_camera = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32)
        
        # FlowNet weights path
        self.pretrained_flownet = '/home/krkavinda/PROJECT-IGI3/FlowNet_models/pytorch/flownets_bn_EPE2.459.pth'
        
        # Trajectories for training and validation
        self.train_trajectories = [

            ('Kite_training/sunny', 'trajectory_0004'),
            ('Kite_training/sunny', 'trajectory_0006'),
            ('Kite_training/sunny', 'trajectory_0012'),
            ('Kite_training/sunny', 'trajectory_0014'),
            
        ]
        
        self.valid_trajectories = [
            ('Kite_training/sunny', 'trajectory_0016'),
            ('Kite_training/sunny', 'trajectory_0001'),

            ('Kite_training/sunny', 'trajectory_0007'),
            ('Kite_training/sunny', 'trajectory_0013'),



        ]
        
        # Output paths
        self.model_dir = './models'
        self.log_dir = './logs'
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.model_name = f"rgbvo_im{self.img_h}x{self.img_w}_s{self.seq_len}_b{self.batch_size}"
        self.model_path = os.path.join(self.model_dir, f"{self.model_name}.pt")
        
# Create global parameters instance
par = Parameters()