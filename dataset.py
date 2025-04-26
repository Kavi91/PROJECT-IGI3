import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import re
import logging
import random

from params import par
from augment import create_vo_augmentation_pipeline
from IMU import IMUPreprocessor

logger = logging.getLogger(__name__)

def natural_sort_key(s):
    """Sort strings with numbers in natural order."""
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', s)]

class VisualInertialOdometryDataset(Dataset):
    """Dataset for visual-inertial odometry."""
    
    def __init__(self, trajectories, is_training=True, use_imu=True, use_integrated_imu=True):
        """
        Args:
            trajectories: List of (climate_set, trajectory_id) tuples
            is_training: Whether this is a training dataset
            use_imu: Whether to use IMU data
            use_integrated_imu: Whether to use integrated IMU features
        """
        self.is_training = is_training
        self.dataset_type = 'train' if is_training else 'val'
        self.trajectories = trajectories
        self.use_imu = use_imu
        self.use_integrated_imu = use_integrated_imu
        
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
                # Customize augmentation parameters here
                color_jitter_prob=0.8,
                brightness_range=0.3,
                contrast_range=0.3,
                saturation_range=0.3,
                hue_range=0.1,
                noise_prob=0.4,
                noise_std=0.03,
                rotation_prob=0.6,
                rotation_max_degrees=3.0,
                perspective_prob=0.3,
                perspective_scale=0.05,
                cutout_prob=0.5,
                cutout_size_range=(0.05, 0.15)
            )
        
        # Dataset info
        self.sequences = []
        self._prepare_sequences()
        
        logger.info(f"Created {self.dataset_type} dataset with {len(self.sequences)} sequences")
        if self.use_imu:
            logger.info(f"Using IMU data with {'integrated' if use_integrated_imu else 'raw'} features")
    
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
                
            # Get IMU file if using IMU
            imu_file = None
            imu_bias_file = None
            if self.use_imu:
                imu_dir = os.path.join(par.data_dir, climate_set, trajectory, "imu")
                if os.path.exists(imu_dir):
                    imu_file = os.path.join(imu_dir, "imu.npy")
                    imu_bias_file = os.path.join(imu_dir, "imu_init_bias.npy")
                    if not os.path.exists(imu_file):
                        logger.warning(f"IMU file not found: {imu_file}")
                        imu_file = None
                    if not os.path.exists(imu_bias_file):
                        logger.warning(f"IMU bias file not found: {imu_bias_file}")
                        imu_bias_file = None
            
            # Load poses
            poses = np.load(pose_file)
            
            # Load IMU data if available
            imu_data = None
            imu_bias = None
            if imu_file is not None:
                try:
                    imu_data = np.load(imu_file)
                    # Check if IMU data length matches
                    if len(imu_data) < len(poses):
                        logger.warning(f"IMU data shorter than poses for {climate_set}/{trajectory}: {len(imu_data)} vs {len(poses)}. Padding...")
                        # Pad with last value
                        pad_width = ((0, len(poses) - len(imu_data)), (0, 0))
                        imu_data = np.pad(imu_data, pad_width, mode='edge')
                    elif len(imu_data) > len(poses):
                        logger.warning(f"IMU data longer than poses for {climate_set}/{trajectory}: {len(imu_data)} vs {len(poses)}. Truncating...")
                        imu_data = imu_data[:len(poses)]
                    
                    # Load IMU bias if available
                    if imu_bias_file is not None:
                        imu_bias = np.load(imu_bias_file)
                        # Check if bias has correct shape [2, 3] (acc_bias, gyro_bias)
                        if imu_bias.shape != (2, 3):
                            logger.warning(f"IMU bias has unexpected shape {imu_bias.shape} for {climate_set}/{trajectory}")
                            imu_bias = None
                except Exception as e:
                    logger.error(f"Error loading IMU data for {climate_set}/{trajectory}: {e}")
                    imu_data = None
                    imu_bias = None
            
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
                
                # Get IMU data for this sequence if available
                seq_imu_data = None
                if imu_data is not None:
                    seq_imu_data = imu_data[i:i+par.seq_len+1]
                
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
                    'imu_data': seq_imu_data,
                    'imu_bias': imu_bias,
                    'climate_set': climate_set,
                    'trajectory': trajectory,
                    'start_idx': i
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
    
    def _preprocess_imu(self, imu_data, imu_bias=None, seq_len=None):
        """
        Preprocess IMU data with bias correction and optional integration.
        
        Args:
            imu_data: Raw IMU data [seq_len, 6]
            imu_bias: IMU bias values [2, 3] (acc_bias, gyro_bias)
            seq_len: Required sequence length for output features
            
        Returns:
            Preprocessed IMU data [seq_len, feature_dim]
        """
        # Skip if no IMU data
        if imu_data is None:
            return None
        
        # Extract bias values
        acc_bias = None
        gyro_bias = None
        if imu_bias is not None:
            acc_bias = imu_bias[0]
            gyro_bias = imu_bias[1]
        
        # Get body_to_camera transformation from params if available
        body_to_camera = None
        if hasattr(par, 'body_to_camera'):
            if isinstance(par.body_to_camera, torch.Tensor):
                body_to_camera = par.body_to_camera.cpu().numpy()
            elif isinstance(par.body_to_camera, np.ndarray):
                body_to_camera = par.body_to_camera
        
        # Create IMU preprocessor
        imu_preprocessor = IMUPreprocessor(
            gyro_bias_init=gyro_bias,
            acc_bias_init=acc_bias,
            apply_integration=self.use_integrated_imu,
            dt=0.01,  # 100Hz
            body_to_camera=body_to_camera
        )
        
        # Process IMU data with explicit sequence length requirement
        try:
            processed_imu = imu_preprocessor.process_sequence(
                imu_data, 
                integrate_segments=self.use_integrated_imu,
                seq_len=seq_len
            )
            return processed_imu
        except Exception as e:
            logger.error(f"Error in IMU preprocessing: {e}")
            
            # Fallback method: simple direct processing
            if imu_data is None or len(imu_data) == 0:
                return None
                
            # Apply simple bias correction
            corrected = imu_data.copy()
            if imu_bias is not None:
                corrected[:, 0:3] -= acc_bias
                corrected[:, 3:6] -= gyro_bias
            
            # Just return the downsampled IMU data to match sequence length
            if seq_len is None:
                return corrected
                
            # Ensure we have enough data for the required sequence length
            if len(corrected) < seq_len:
                # Pad by repeating the last frame
                if len(corrected) > 0:
                    last_frame = corrected[-1:]
                    padding = np.repeat(last_frame, seq_len - len(corrected), axis=0)
                    corrected = np.concatenate([corrected, padding], axis=0)
                else:
                    # Create zeros if no IMU data
                    corrected = np.zeros((seq_len, 6))
            else:
                # Downsample if we have more data than needed
                indices = np.linspace(0, len(corrected)-1, seq_len, dtype=int)
                corrected = corrected[indices]
                
            # For integrated features, create a simplified version
            if self.use_integrated_imu:
                # Extract basic components and pad with zeros
                feature_dim = 21  # 3 (acc) + 3 (gyro) + 9 (orientation) + 3 (velocity) + 3 (position)
                features = np.zeros((seq_len, feature_dim))
                
                # Use actual IMU values for acc and gyro, zeros for derived features
                features[:, 0:6] = corrected[:, 0:6]  # acc and gyro
                
                return features
            else:
                # Return raw corrected data
                return corrected
    
    def __len__(self):
        """Return the number of sequences in the dataset."""
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """Get a sequence by index."""
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
        
        # Get IMU data with explicit sequence length requirement
        imu_data = None
        if self.use_imu and sequence['imu_data'] is not None:
            # Debug IMU data shape for first few sequences
            # if idx < 3:
            #     logger.info(f"Original IMU data shape for idx {idx}: {sequence['imu_data'].shape}")
            #     logger.info(f"Sequence length needed: {par.seq_len}")
            
            # Preprocess IMU data with explicit sequence length requirement
            try:
                processed_imu = self._preprocess_imu(sequence['imu_data'], sequence['imu_bias'], par.seq_len)
                
                # Check if we have valid IMU features
                if processed_imu is not None:
                    if processed_imu.shape[0] < par.seq_len:
                        # This should not happen with our fixed processor but handle it just in case
                        logger.warning(f"Processed IMU still has wrong length: {processed_imu.shape[0]} vs {par.seq_len} needed")
                        
                        # Pad with last row repeated
                        if processed_imu.shape[0] > 0:
                            last_row = processed_imu[-1:]
                            padding = np.repeat(last_row, par.seq_len - processed_imu.shape[0], axis=0)
                            processed_imu = np.concatenate([processed_imu, padding], axis=0)
                        else:
                            # Create zeros array of correct shape if no valid data
                            feature_dim = 21 if self.use_integrated_imu else 6
                            processed_imu = np.zeros((par.seq_len, feature_dim))
                    
                    # Convert to tensor
                    imu_data = torch.FloatTensor(processed_imu)
                    
                    # Debug first few frames
                    # if idx < 3:
                    #     logger.info(f"Processed IMU shape: {imu_data.shape}")
                    #     logger.info(f"IMU data sample[0]: {imu_data[0][:5]}...")
            except Exception as e:
                logger.error(f"Error processing IMU data for sequence {idx}: {e}")
                # Create a fallback tensor of zeros with correct shape
                feature_dim = 21 if self.use_integrated_imu else 6
                imu_data = torch.zeros((par.seq_len, feature_dim))
        
        # Apply augmentations during training
        if self.is_training:
            # Apply sequence-consistent augmentations to RGB only
            rgb_sequence = self.augmentation_pipeline(rgb_sequence)
            
            # Note: We don't augment IMU data to maintain physical consistency
        
        # Apply FlowNet normalization if required
        if par.minus_point_5:
            rgb_sequence = rgb_sequence - 0.5
        
        # Apply normalization
        for i in range(rgb_sequence.size(0)):
            rgb_sequence[i] = self.normalizer(rgb_sequence[i])
        
        return rgb_sequence, rel_poses, abs_poses, imu_data

def collate_with_imu(batch):
    """
    Custom collate function to handle variable-length IMU data.
    
    Args:
        batch: List of (rgb_sequence, rel_poses, abs_poses, imu_data) tuples
        
    Returns:
        Batched tensors with imu_data as None if any sample has None
    """
    # Unzip the batch
    rgb_sequences, rel_poses, abs_poses, imu_data = zip(*batch)
    
    # Stack RGB, rel_poses, and abs_poses
    rgb_sequences = torch.stack(rgb_sequences)
    rel_poses = torch.stack(rel_poses)
    abs_poses = torch.stack(abs_poses)
    
    # Check if all IMU data is available
    if all(imu is not None for imu in imu_data):
        imu_data = torch.stack(imu_data)
    else:
        imu_data = None
    
    return rgb_sequences, rel_poses, abs_poses, imu_data

def create_data_loaders(batch_size=None, use_imu=True, use_integrated_imu=True):
    """Create data loaders for training and validation."""
    if batch_size is None:
        batch_size = par.batch_size
    
    # Create datasets
    train_dataset = VisualInertialOdometryDataset(
        par.train_trajectories, 
        is_training=True, 
        use_imu=use_imu,
        use_integrated_imu=use_integrated_imu
    )
    
    val_dataset = VisualInertialOdometryDataset(
        par.valid_trajectories, 
        is_training=False, 
        use_imu=use_imu,
        use_integrated_imu=use_integrated_imu
    )
    
    # Create data loaders with custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=par.n_processors,
        pin_memory=par.pin_mem,
        drop_last=True,
        collate_fn=collate_with_imu
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=par.n_processors,
        pin_memory=par.pin_mem,
        drop_last=False,
        collate_fn=collate_with_imu
    )
    
    return train_loader, val_loader