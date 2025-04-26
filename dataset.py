import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import re
import logging

from params import par
from augment import create_vo_augmentation_pipeline

logger = logging.getLogger(__name__)

def natural_sort_key(s):
    """Sort strings with numbers in natural order."""
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', s)]

class RGBOdometryDataset(Dataset):
    """Dataset for RGB-based visual odometry."""
    
    def __init__(self, trajectories, is_training=True):
        """
        Args:
            trajectories: List of (climate_set, trajectory_id) tuples
            is_training: Whether this is a training dataset
        """
        self.is_training = is_training
        self.dataset_type = 'train' if is_training else 'val'
        self.trajectories = trajectories
        
        # Image transformations
        transform_ops = [
            transforms.Resize((par.img_h, par.img_w)),
            transforms.ToTensor()
        ]
        self.transformer = transforms.Compose(transform_ops)
        self.normalizer = transforms.Normalize(mean=par.img_means_rgb, std=par.img_stds_rgb)
        
        # Create augmentation pipeline for training
        if is_training:
            self.augmentation_pipeline = create_vo_augmentation_pipeline(
                # Customize augmentation parameters here if needed
                brightness_range=0.3,
                contrast_range=0.3,
                saturation_range=0.3,
                noise_prob=0.4,
                rotation_max_degrees=2.0,
                perspective_scale=0.05,
                cutout_prob=0.4,
                cutout_size_range=(0.05, 0.15),
                fog_prob=0.1,
                streak_prob=0.1
            )
        
        # Dataset info
        self.sequences = []
        self._prepare_sequences()
        
        logger.info(f"Created {self.dataset_type} dataset with {len(self.sequences)} sequences")
    
    def _prepare_sequences(self):
        """Prepare sequences from trajectories."""
        for climate_set, trajectory in self.trajectories:
            # Get RGB image paths
            rgb_dir = os.path.join(par.image_dir, climate_set, trajectory, "image_rgb")
            if not os.path.exists(rgb_dir):
                logger.warning(f"RGB directory not found: {rgb_dir}")
                continue
                
            # Get pose file
            pose_file = os.path.join(par.pose_dir, climate_set, "poses", f"poses_{trajectory.split('_')[1]}.npy")
            if not os.path.exists(pose_file):
                logger.warning(f"Pose file not found: {pose_file}")
                continue
                
            # Load poses
            poses = np.load(pose_file)
            
            # Get RGB images
            rgb_files = [f for f in os.listdir(rgb_dir) if f.lower().endswith('.jpeg')]
            rgb_files.sort(key=natural_sort_key)
            rgb_paths = [os.path.join(rgb_dir, f) for f in rgb_files]
            
            # Check if enough frames
            n_frames = min(len(poses), len(rgb_paths))
            if n_frames < par.seq_len + 1:  # Need at least seq_len + 1 frames
                logger.warning(f"Not enough frames in {climate_set}/{trajectory}: {n_frames}")
                continue
                
            # Create sequences
            for i in range(0, n_frames - par.seq_len):
                seq_rgb_paths = rgb_paths[i:i+par.seq_len+1]
                seq_poses = poses[i:i+par.seq_len+1]
                
                # Normalize sequence poses relative to the first pose
                seq_poses = seq_poses - seq_poses[0]
                
                # Compute relative poses (from frame t to t+1)
                rel_poses = []
                for j in range(par.seq_len):
                    pose_t = seq_poses[j]
                    pose_t_plus_1 = seq_poses[j+1]
                    rel_pose = self._compute_relative_pose(pose_t, pose_t_plus_1)
                    rel_poses.append(rel_pose)
                
                # Store sequence data
                self.sequences.append({
                    'rgb_paths': seq_rgb_paths,
                    'rel_poses': np.array(rel_poses),
                    'abs_poses': seq_poses,
                    'climate_set': climate_set,
                    'trajectory': trajectory
                })
    
    def _compute_relative_pose(self, pose_t, pose_t_plus_1):
        """
        Compute relative pose from pose_t to pose_t_plus_1.
        Each pose is [roll, pitch, yaw, x, y, z].
        """
        from helper import euler_to_rotation_matrix, rotation_matrix_to_euler
        
        # Extract rotations and translations
        R_t = euler_to_rotation_matrix(pose_t[:3])
        R_t_plus_1 = euler_to_rotation_matrix(pose_t_plus_1[:3])
        t_t = pose_t[3:6]
        t_t_plus_1 = pose_t_plus_1[3:6]
        
        # Compute relative rotation
        R_rel = R_t.T @ R_t_plus_1
        euler_rel = rotation_matrix_to_euler(R_rel)
        
        # Compute relative translation in the coordinate frame of pose_t
        t_rel = R_t.T @ (t_t_plus_1 - t_t)
        
        # Combine relative pose
        rel_pose = np.concatenate([euler_rel, t_rel])
        return rel_pose
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        # Load RGB images
        rgb_sequence = []
        for img_path in sequence['rgb_paths']:
            try:
                # Open and transform image
                img = Image.open(img_path)
                img_tensor = self.transformer(img)
                rgb_sequence.append(img_tensor)
            except Exception as e:
                logger.error(f"Error loading image {img_path}: {e}")
                # Create a zero tensor on error
                img_tensor = torch.zeros(3, par.img_h, par.img_w)
                rgb_sequence.append(img_tensor)
        
        # Stack images
        rgb_sequence = torch.stack(rgb_sequence)  # [seq_len+1, 3, H, W]
        
        # Get poses
        rel_poses = torch.FloatTensor(sequence['rel_poses'])
        abs_poses = torch.FloatTensor(sequence['abs_poses'])
        
        # Apply augmentations during training
        if self.is_training:
            # Apply sequence-consistent augmentations
            rgb_sequence = self.augmentation_pipeline(rgb_sequence)
        
        # Apply FlowNet normalization if required
        if par.minus_point_5:
            rgb_sequence = rgb_sequence - 0.5
        
        # Apply normalization
        for i in range(rgb_sequence.size(0)):
            rgb_sequence[i] = self.normalizer(rgb_sequence[i])
        
        return rgb_sequence, rel_poses, abs_poses

def create_data_loaders(batch_size=None):
    """Create data loaders for training and validation."""
    if batch_size is None:
        batch_size = par.batch_size
    
    # Create datasets
    train_dataset = RGBOdometryDataset(par.train_trajectories, is_training=True)
    val_dataset = RGBOdometryDataset(par.valid_trajectories, is_training=False)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=par.n_processors,
        pin_memory=par.pin_mem,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=par.n_processors,
        pin_memory=par.pin_mem,
        drop_last=False
    )
    
    return train_loader, val_loader