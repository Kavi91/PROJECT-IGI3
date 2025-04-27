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

def open_float16(image):
    """Decode 16-bit float depth map from PNG, as per MidAir official method."""
    try:
        # Use np.array() instead of np.asarray() to ensure a writeable copy
        img = np.array(image, np.uint16)
        img.dtype = np.float16
        return img
    except Exception as e:
        logger.error(f"Error decoding depth image: {e}")
        return None

class VisualInertialOdometryDataset(Dataset):
    """Dataset for visual-inertial odometry with GPS support."""
    
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
        self.use_depth = par.use_depth
        self.use_gps = par.use_gps
        
        # Image transformations for RGB
        transform_ops = [
            transforms.Resize((par.img_h, par.img_w)),
            transforms.ToTensor()
        ]
        self.transformer = transforms.Compose(transform_ops)
        self.normalizer = transforms.Normalize(mean=par.img_means_rgb, std=par.img_stds_rgb)
        
        # Depth transformation (only resizing, ToTensor is applied later)
        self.depth_transform = transforms.Compose([
            transforms.Resize((par.img_h, par.img_w))
        ])
        
        # Depth normalization parameters
        self.depth_mean = par.depth_mean
        self.depth_std = par.depth_std if par.depth_std != 0 else 1.0  # Avoid division by zero
        
        # Create augmentation pipeline for training
        if is_training:
            self.augmentation_pipeline = create_vo_augmentation_pipeline(
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
        if self.use_depth:
            logger.info(f"Using Depth data with mean={self.depth_mean:.4f}, std={self.depth_std:.4f}")
        if self.use_gps:
            logger.info("Using GPS data")
    
    def _prepare_sequences(self):
        """Prepare sequences from trajectories."""
        for climate_set, trajectory in self.trajectories:
            # Get RGB image paths
            rgb_dir = os.path.join(par.image_dir, climate_set, trajectory, "image_rgb")
            if not os.path.exists(rgb_dir):
                logger.warning(f"RGB directory not found: {rgb_dir}")
                continue
                
            # Get pose file - decide whether to use camera frame or body frame
            if hasattr(par, 'use_camera_frame') and par.use_camera_frame:
                pose_file = os.path.join(par.pose_dir, climate_set, "poses", f"camera_poses_{trajectory.split('_')[1]}.npy")
                if not os.path.exists(pose_file):
                    logger.warning(f"Camera frame pose file not found: {pose_file}, falling back to body frame")
                    pose_file = os.path.join(par.pose_dir, climate_set, "poses", f"poses_{trajectory.split('_')[1]}.npy")
            else:
                pose_file = os.path.join(par.pose_dir, climate_set, "poses", f"poses_{trajectory.split('_')[1]}.npy")
                
            if not os.path.exists(pose_file):
                logger.warning(f"Pose file not found: {pose_file}")
                continue
                
            # Get Depth file paths if enabled
            depth_files = []
            depth_paths = []
            if self.use_depth:
                depth_dir = os.path.join(par.image_dir, climate_set, trajectory, "depth")
                if os.path.exists(depth_dir):
                    depth_files = [f for f in os.listdir(depth_dir) if f.lower().endswith('.png')]
                    depth_files.sort(key=natural_sort_key)
                    depth_paths = [os.path.join(depth_dir, f) for f in depth_files]
                else:
                    logger.warning(f"Depth directory not found: {depth_dir}")
                    depth_paths = []
            
            # Get GPS file if enabled
            gps_file = None
            if self.use_gps:
                gps_dir = os.path.join(par.data_dir, climate_set, trajectory, "gps")
                if os.path.exists(gps_dir):
                    if hasattr(par, 'use_camera_frame') and par.use_camera_frame:
                        gps_file = os.path.join(gps_dir, "camera_gps.npy")
                        if not os.path.exists(gps_file):
                            logger.warning(f"Camera GPS file not found: {gps_file}, falling back to body frame")
                            gps_file = os.path.join(gps_dir, "gps.npy")
                    else:
                        gps_file = os.path.join(gps_dir, "gps.npy")
                    
                    if not os.path.exists(gps_file):
                        logger.warning(f"GPS file not found: {gps_file}")
                        gps_file = None
            
            # Get IMU file if using IMU
            imu_file = None
            imu_bias_file = None
            if self.use_imu:
                imu_dir = os.path.join(par.data_dir, climate_set, trajectory, "imu")
                if os.path.exists(imu_dir):
                    if hasattr(par, 'use_camera_frame') and par.use_camera_frame:
                        imu_file = os.path.join(imu_dir, "camera_imu.npy")
                        if not os.path.exists(imu_file):
                            logger.warning(f"Camera IMU file not found: {imu_file}, falling back to body frame")
                            imu_file = os.path.join(imu_dir, "imu.npy")
                    else:
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
            
            # Load GPS data if available
            gps_data = None
            if gps_file is not None:
                try:
                    gps_data = np.load(gps_file)
                    if len(gps_data) < len(poses):
                        logger.warning(f"GPS data shorter than poses for {climate_set}/{trajectory}: {len(gps_data)} vs {len(poses)}. Padding...")
                        pad_width = ((0, len(poses) - len(gps_data)), (0, 0))
                        gps_data = np.pad(gps_data, pad_width, mode='edge')
                    elif len(gps_data) > len(poses):
                        logger.warning(f"GPS data longer than poses for {climate_set}/{trajectory}: {len(gps_data)} vs {len(poses)}. Truncating...")
                        gps_data = gps_data[:len(poses)]
                except Exception as e:
                    logger.error(f"Error loading GPS data for {climate_set}/{trajectory}: {e}")
                    gps_data = None
            
            # Load IMU data if available
            imu_data = None
            imu_bias = None
            if imu_file is not None:
                try:
                    imu_data = np.load(imu_file)
                    if len(imu_data) < len(poses):
                        logger.warning(f"IMU data shorter than poses for {climate_set}/{trajectory}: {len(imu_data)} vs {len(poses)}. Padding...")
                        pad_width = ((0, len(poses) - len(imu_data)), (0, 0))
                        imu_data = np.pad(imu_data, pad_width, mode='edge')
                    elif len(imu_data) > len(poses):
                        logger.warning(f"IMU data longer than poses for {climate_set}/{trajectory}: {len(imu_data)} vs {len(poses)}. Truncating...")
                        imu_data = imu_data[:len(poses)]
                    
                    if imu_bias_file is not None:
                        imu_bias = np.load(imu_bias_file)
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
            
            # Determine number of frames
            n_frames = min(len(poses), len(rgb_paths))
            if self.use_depth and depth_paths:
                n_frames = min(n_frames, len(depth_paths))
            if n_frames < par.seq_len + 1:
                logger.warning(f"Not enough frames in {climate_set}/{trajectory}: {n_frames}")
                continue
                
            # Create sequences
            for i in range(0, n_frames - par.seq_len):
                seq_rgb_paths = rgb_paths[i:i+par.seq_len+1]
                seq_poses = poses[i:i+par.seq_len+1]
                
                # Get Depth data for this sequence if available
                seq_depth_paths = []
                if self.use_depth and depth_paths:
                    seq_depth_paths = depth_paths[i:i+par.seq_len+1]
                
                # Get GPS data for this sequence if available
                seq_gps_data = None
                if gps_data is not None:
                    seq_gps_data = gps_data[i:i+par.seq_len+1]
                
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
                    'depth_paths': seq_depth_paths,
                    'rel_poses': np.array(rel_poses),
                    'abs_poses': seq_poses,
                    'imu_data': seq_imu_data,
                    'gps_data': seq_gps_data,
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
        if imu_data is None or len(imu_data) < 4:
            return None
        
        # Extract bias values
        acc_bias = None
        gyro_bias = None
        if imu_bias is not None:
            acc_bias = imu_bias[0]
            gyro_bias = imu_bias[1]
        
        # Create IMU preprocessor
        imu_preprocessor = IMUPreprocessor(
            gyro_bias_init=gyro_bias,
            acc_bias_init=acc_bias,
            apply_integration=self.use_integrated_imu,
            dt=0.01,
            body_to_camera=None,  # Don't apply transformation here - data already in correct frame
            body_to_camera_translation=None
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
                    corrected = np.zeros((seq_len, 6))
            else:
                indices = np.linspace(0, len(corrected)-1, seq_len, dtype=int)
                corrected = corrected[indices]
                
            # For integrated features, create a simplified version
            if self.use_integrated_imu:
                feature_dim = 21
                features = np.zeros((seq_len, feature_dim))
                features[:, 0:6] = corrected[:, 0:6]
                return features
            else:
                return corrected
    
    def _preprocess_gps(self, gps_data, seq_len=None):
        """
        Preprocess GPS data to extract relevant features.
        
        Args:
            gps_data: Raw GPS data [seq_len, 9] (position [x, y, z], velocity [vx, vy, vz], signal [num_sats, GDOP, PDOP])
            seq_len: Required sequence length for output features
            
        Returns:
            Preprocessed GPS data [seq_len, feature_dim]
        """
        if gps_data is None or len(gps_data) == 0:
            return None
            
        # Extract features: position, velocity, and signal info
        processed_gps = gps_data.copy()
        
        # Ensure data matches the required sequence length
        if seq_len is None:
            return processed_gps
            
        if len(processed_gps) < seq_len:
            # Pad by repeating the last frame
            if len(processed_gps) > 0:
                last_frame = processed_gps[-1:]
                padding = np.repeat(last_frame, seq_len - len(processed_gps), axis=0)
                processed_gps = np.concatenate([processed_gps, padding], axis=0)
            else:
                processed_gps = np.zeros((seq_len, 9))
        else:
            indices = np.linspace(0, len(processed_gps)-1, seq_len, dtype=int)
            processed_gps = processed_gps[indices]
            
        return processed_gps
    
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
                img = Image.open(img_path)
                img_tensor = self.transformer(img)
                rgb_sequence.append(img_tensor)
            except Exception as e:
                logger.error(f"Error loading image {img_path}: {e}")
                img_tensor = torch.zeros(3, par.img_h, par.img_w)
                rgb_sequence.append(img_tensor)
        
        rgb_sequence = torch.stack(rgb_sequence)  # [seq_len+1, 3, H, W]
        
        # Load Depth images if enabled (using official MidAir decoding)
        depth_sequence = None
        if self.use_depth and sequence['depth_paths']:
            depth_sequence = []
            for depth_path in sequence['depth_paths']:
                try:
                    # Load depth image as PIL Image
                    depth_img = Image.open(depth_path)
                    # Decode depth using official method
                    depth_array = open_float16(depth_img)
                    if depth_array is None:
                        raise ValueError("Failed to decode depth image")
                    
                    # Process depth as per official script
                    depth_array = np.clip(depth_array, 1, 1250, depth_array)  # Clip between 1 and 1250 meters
                    # Apply logarithmic normalization: (log(depth) - 1) / (log(1250) - 1)
                    depth_array = (np.log(depth_array) - np.log(1)) / (np.log(1250) - np.log(1))
                    # Apply standardization using depth_mean and depth_std
                    depth_array = (depth_array - self.depth_mean) / self.depth_std
                    # Convert to PIL Image for transformation
                    depth_img = Image.fromarray(depth_array.astype(np.float32))
                    # Apply resizing
                    depth_img = self.depth_transform(depth_img)
                    # Convert to tensor
                    depth_tensor = transforms.ToTensor()(depth_img)
                    depth_sequence.append(depth_tensor)
                except Exception as e:
                    logger.error(f"Error loading depth image {depth_path}: {e}")
                    depth_tensor = torch.zeros(1, par.img_h, par.img_w)
                    depth_sequence.append(depth_tensor)
            depth_sequence = torch.stack(depth_sequence)  # [seq_len+1, 1, H, W]
        
        # Get poses
        rel_poses = torch.FloatTensor(sequence['rel_poses'])
        abs_poses = torch.FloatTensor(sequence['abs_poses'])
        
        # Get GPS data if enabled
        gps_data = None
        if self.use_gps and sequence['gps_data'] is not None:
            try:
                processed_gps = self._preprocess_gps(sequence['gps_data'], par.seq_len)
                if processed_gps is not None:
                    if processed_gps.shape[0] < par.seq_len:
                        logger.warning(f"Processed GPS still has wrong length: {processed_gps.shape[0]} vs {par.seq_len} needed")
                        if processed_gps.shape[0] > 0:
                            last_row = processed_gps[-1:]
                            padding = np.repeat(last_row, par.seq_len - processed_gps.shape[0], axis=0)
                            processed_gps = np.concatenate([processed_gps, padding], axis=0)
                        else:
                            processed_gps = np.zeros((par.seq_len, 9))
                    gps_data = torch.FloatTensor(processed_gps)
            except Exception as e:
                logger.error(f"Error processing GPS data for sequence {idx}: {e}")
                gps_data = torch.zeros((par.seq_len, 9))
        
        # Get IMU data with explicit sequence length requirement
        imu_data = None
        if self.use_imu and sequence['imu_data'] is not None:
            try:
                processed_imu = self._preprocess_imu(sequence['imu_data'], sequence['imu_bias'], par.seq_len)
                if processed_imu is not None:
                    if processed_imu.shape[0] < par.seq_len:
                        logger.warning(f"Processed IMU still has wrong length: {processed_imu.shape[0]} vs {par.seq_len} needed")
                        if processed_imu.shape[0] > 0:
                            last_row = processed_imu[-1:]
                            padding = np.repeat(last_row, par.seq_len - processed_imu.shape[0], axis=0)
                            processed_imu = np.concatenate([processed_imu, padding], axis=0)
                        else:
                            feature_dim = 21 if self.use_integrated_imu else 6
                            processed_imu = np.zeros((par.seq_len, feature_dim))
                    imu_data = torch.FloatTensor(processed_imu)
            except Exception as e:
                logger.error(f"Error processing IMU data for sequence {idx}: {e}")
                feature_dim = 21 if self.use_integrated_imu else 6
                imu_data = torch.zeros((par.seq_len, feature_dim))
        
        # Apply augmentations during training
        # if self.is_training:
        #     rgb_sequence = self.augmentation_pipeline(rgb_sequence)
        #     if self.use_depth and depth_sequence is not None:
        #          # Apply geometric augmentations to depth (skip color jitter/noise)
        #         geometric_augs = create_vo_augmentation_pipeline(
        #         color_jitter_prob=0.8,
        #         brightness_range=0.3,
        #         contrast_range=0.3,
        #         saturation_range=0.3,
        #         hue_range=0.1,
        #         rotation_prob=0.6,
        #         rotation_max_degrees=3.0,
        #         perspective_prob=0.3,
        #         perspective_scale=0.05,
        #         cutout_prob=0.5,
        #         cutout_size_range=(0.05, 0.15)
        #     )
            # Note: We don't augment IMU or GPS data to maintain physical consistency
        
        # Apply FlowNet normalization if required
        if par.minus_point_5:
            rgb_sequence = rgb_sequence - 0.5
        
        # Apply normalization to RGB
        for i in range(rgb_sequence.size(0)):
            rgb_sequence[i] = self.normalizer(rgb_sequence[i])
        
        # Return data
        return_items = [rgb_sequence]
        if self.use_depth:
            return_items.append(depth_sequence if depth_sequence is not None else torch.zeros(par.seq_len+1, 1, par.img_h, par.img_w))
        if self.use_imu:
            return_items.append(imu_data if imu_data is not None else torch.zeros(par.seq_len, 21 if self.use_integrated_imu else 6))
        if self.use_gps:
            return_items.append(gps_data if gps_data is not None else torch.zeros(par.seq_len, 9))
        return_items.extend([rel_poses, abs_poses])
        return tuple(return_items)

def collate_with_imu(batch):
    """
    Custom collate function to handle variable-length data with GPS support.
    
    Args:
        batch: List of (rgb_sequence, [depth_sequence], [imu_data], [gps_data], rel_poses, abs_poses) tuples
        
    Returns:
        Batched tensors with optional modalities as None if any sample has None
    """
    # Unzip the batch based on enabled modalities
    idx = 0
    rgb_sequences = [item[idx] for item in batch]
    idx += 1
    
    depth_sequences = None
    if par.use_depth:
        depth_sequences = [item[idx] for item in batch]
        idx += 1
    
    imu_data = None
    if par.use_imu:
        imu_data = [item[idx] for item in batch]
        idx += 1
    
    gps_data = None
    if par.use_gps:
        gps_data = [item[idx] for item in batch]
        idx += 1
    
    rel_poses = [item[idx] for item in batch]
    abs_poses = [item[idx+1] for item in batch]
    
    # Stack RGB, rel_poses, and abs_poses
    rgb_sequences = torch.stack(rgb_sequences)
    rel_poses = torch.stack(rel_poses)
    abs_poses = torch.stack(abs_poses)
    
    # Stack Depth if enabled
    if par.use_depth:
        depth_sequences = torch.stack(depth_sequences) if all(d is not None for d in depth_sequences) else None
    
    # Stack IMU if enabled
    if par.use_imu:
        imu_data = torch.stack(imu_data) if all(imu is not None for imu in imu_data) else None
    
    # Stack GPS if enabled
    if par.use_gps:
        gps_data = torch.stack(gps_data) if all(gps is not None for gps in gps_data) else None
    
    return_items = [rgb_sequences]
    if par.use_depth:
        return_items.append(depth_sequences)
    if par.use_imu:
        return_items.append(imu_data)
    if par.use_gps:
        return_items.append(gps_data)
    return_items.extend([rel_poses, abs_poses])
    return tuple(return_items)

def create_data_loaders(batch_size=None, use_imu=True, use_integrated_imu=True, use_depth=None):
    """Create data loaders for training and validation."""
    if batch_size is None:
        batch_size = par.batch_size
    
    # Allow overriding depth parameter
    actual_use_depth = par.use_depth if use_depth is None else use_depth
    
    # Create datasets
    train_dataset = VisualInertialOdometryDataset(
        par.train_trajectories, 
        is_training=True, 
        use_imu=use_imu,
        use_integrated_imu=use_integrated_imu
    )
    
    # Store the depth flag in train_dataset
    train_dataset.use_depth = actual_use_depth
    
    val_dataset = VisualInertialOdometryDataset(
        par.valid_trajectories, 
        is_training=False, 
        use_imu=use_imu,
        use_integrated_imu=use_integrated_imu
    )
    
    # Store the depth flag in val_dataset
    val_dataset.use_depth = actual_use_depth
    
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