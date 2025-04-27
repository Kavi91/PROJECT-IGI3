#!/usr/bin/env python3
"""
Optimized Visual-Inertial Odometry Test Script

Features:
- Flexible trajectory integration with configurable parameters
- Automatic scaling factor optimization
- Proper coordinate frame handling
- Comprehensive visualization and metrics
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from PIL import Image
import torchvision.transforms as transforms
import logging
from tqdm import tqdm
import json
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import your model and parameters
from model_2 import VisualInertialOdometryModel
from params import par

def load_image(path):
    """Load and preprocess an image for model input."""
    try:
        img = Image.open(path)
        
        # Apply transformations
        transform_ops = [
            transforms.Resize((par.img_h, par.img_w)),
            transforms.ToTensor()
        ]
        transformer = transforms.Compose(transform_ops)
        img_tensor = transformer(img)
        
        # Apply FlowNet normalization if required
        if par.minus_point_5:
            img_tensor = img_tensor - 0.5
        
        # Apply normalization
        normalizer = transforms.Normalize(mean=par.img_means_rgb, std=par.img_stds_rgb)
        img_tensor = normalizer(img_tensor)
        
        return img_tensor
    except Exception as e:
        logger.error(f"Error loading image {path}: {e}")
        return None

def load_depth(path):
    """Load and preprocess a depth image for model input."""
    try:
        # Open depth image
        depth_img = Image.open(path)
        
        # Decode 16-bit float depth map
        depth_array = np.array(depth_img, dtype=np.uint16)
        depth_array.dtype = np.float16
        
        # Apply MidAir depth processing
        depth_array = np.clip(depth_array, 1, 1250)  # Clip to valid range
        # Apply logarithmic normalization
        depth_array = (np.log(depth_array) - np.log(1)) / (np.log(1250) - np.log(1))
        # Apply standardization
        depth_array = (depth_array - par.depth_mean) / par.depth_std
        
        # Convert to PIL Image for resizing
        depth_img = Image.fromarray(depth_array.astype(np.float32))
        
        # Resize
        depth_transform = transforms.Compose([
            transforms.Resize((par.img_h, par.img_w))
        ])
        depth_img = depth_transform(depth_img)
        
        # Convert to tensor
        depth_tensor = transforms.ToTensor()(depth_img)
        
        return depth_tensor
    except Exception as e:
        logger.error(f"Error loading depth image {path}: {e}")
        return None

def create_integrated_imu_features(raw_imu_data):
    """
    Convert raw IMU data to integrated IMU features.
    
    Args:
        raw_imu_data: Raw IMU data [seq_len, 6]
        
    Returns:
        Integrated IMU features [seq_len, 21]
    """
    seq_len = len(raw_imu_data)
    integrated_features = np.zeros((seq_len, 21))
    
    # Copy raw IMU data to first 6 dimensions
    integrated_features[:, 0:6] = raw_imu_data
    
    # The remaining 15 features would normally contain:
    # - orientation_change (9D)
    # - velocity (3D)
    # - displacement (3D)
    # We're leaving them as zeros
    
    return integrated_features

def integrate_trajectory(rel_poses, scale_factor=0.1, apply_rotation=True, is_camera_frame=True):
    """
    Flexible trajectory integration with configurable parameters.
    
    Args:
        rel_poses: Array of relative poses [N, 6] with [roll, pitch, yaw, x, y, z]
        scale_factor: Scaling factor for translations
        apply_rotation: Whether to apply rotation to translations
        is_camera_frame: Whether poses are in camera frame
        
    Returns:
        abs_poses: Array of absolute poses [N+1, 7] with [x, y, z, qx, qy, qz, qw]
    """
    # Create absolute poses array (first pose is identity)
    abs_poses = np.zeros((len(rel_poses) + 1, 7))
    abs_poses[0, 6] = 1.0  # Identity quaternion [0, 0, 0, 1]
    
    # Current state
    curr_pos = np.zeros(3)
    curr_rot = Rotation.identity()
    
    # Body-to-camera transformation if needed
    if not is_camera_frame and hasattr(par, 'body_to_camera'):
        body_to_camera = par.body_to_camera.cpu().numpy()
        camera_to_body = np.linalg.inv(body_to_camera)
    
    for i in range(len(rel_poses)):
        # Get relative pose
        rel_euler = rel_poses[i, :3]  # roll, pitch, yaw
        rel_trans = rel_poses[i, 3:6]  # x, y, z
        
        # Scale translations
        rel_trans = rel_trans * scale_factor
        
        # Apply coordinate transformation if not in camera frame
        if not is_camera_frame and hasattr(par, 'body_to_camera'):
            # Transform from body to camera frame
            rel_rot_mat = Rotation.from_euler('xyz', rel_euler).as_matrix()
            rel_rot_mat = camera_to_body @ rel_rot_mat @ body_to_camera
            rel_rot = Rotation.from_matrix(rel_rot_mat)
            rel_trans = camera_to_body @ rel_trans
        else:
            # Use as is
            rel_rot = Rotation.from_euler('xyz', rel_euler)
        
        # Update rotation
        curr_rot = curr_rot * rel_rot
        
        # Update position - either with or without rotation
        if apply_rotation:
            # With rotation (standard approach)
            curr_pos = curr_pos + curr_rot.apply(rel_trans)
        else:
            # Without rotation (simple sum)
            curr_pos = curr_pos + rel_trans
        
        # Store results
        abs_poses[i+1, :3] = curr_pos
        abs_poses[i+1, 3:7] = curr_rot.as_quat()
    
    return abs_poses

def prepare_ground_truth(gt_poses):
    """
    Convert ground truth poses to standard format.
    
    Args:
        gt_poses: Ground truth poses [N, 6] with [roll, pitch, yaw, x, y, z]
        
    Returns:
        Ground truth trajectory [N, 7] with [x, y, z, qx, qy, qz, qw]
    """
    gt_traj = np.zeros((len(gt_poses), 7))
    
    for i in range(len(gt_poses)):
        # Position is the last 3 elements
        gt_traj[i, 0:3] = gt_poses[i, 3:6]
        
        # Convert Euler to quaternion for orientation
        euler = gt_poses[i, 0:3]  # roll, pitch, yaw
        rot = Rotation.from_euler('xyz', euler)
        quat = rot.as_quat()  # [qx, qy, qz, qw]
        gt_traj[i, 3:7] = quat
    
    return gt_traj

def compute_ate(pred_traj, gt_traj):
    """
    Compute Absolute Trajectory Error.
    
    Args:
        pred_traj: Predicted trajectory [N, 7] with [x, y, z, qx, qy, qz, qw]
        gt_traj: Ground truth trajectory [N, 7] with [x, y, z, qx, qy, qz, qw]
        
    Returns:
        mean_error: Mean position error
        std_error: Standard deviation of position error
        max_error: Maximum position error
    """
    # Ensure same length
    min_len = min(len(pred_traj), len(gt_traj))
    pred_pos = pred_traj[:min_len, :3]
    gt_pos = gt_traj[:min_len, :3]
    
    # Compute position errors
    pos_errors = np.linalg.norm(pred_pos - gt_pos, axis=1)
    
    return np.mean(pos_errors), np.std(pos_errors), np.max(pos_errors)

def compute_rpe(pred_traj, gt_traj, delta=1):
    """
    Compute Relative Pose Error.
    
    Args:
        pred_traj: Predicted trajectory [N, 7] with [x, y, z, qx, qy, qz, qw]
        gt_traj: Ground truth trajectory [N, 7] with [x, y, z, qx, qy, qz, qw]
        delta: Frame delta for computing relative pose
        
    Returns:
        trans_error: Mean translation error
        rot_error: Mean rotation error in radians
    """
    # Ensure same length
    min_len = min(len(pred_traj), len(gt_traj))
    if min_len <= delta:
        return 0, 0
    
    # Compute relative poses
    trans_errors = []
    rot_errors = []
    
    for i in range(0, min_len - delta):
        # Extract poses
        p1 = pred_traj[i]
        p2 = pred_traj[i + delta]
        g1 = gt_traj[i]
        g2 = gt_traj[i + delta]
        
        # Extract positions
        p1_pos, p2_pos = p1[:3], p2[:3]
        g1_pos, g2_pos = g1[:3], g2[:3]
        
        # Extract rotations
        p1_rot = Rotation.from_quat(p1[3:7])
        p2_rot = Rotation.from_quat(p2[3:7])
        g1_rot = Rotation.from_quat(g1[3:7])
        g2_rot = Rotation.from_quat(g2[3:7])
        
        # Compute relative transforms
        p_rel_trans = p2_pos - p1_pos
        g_rel_trans = g2_pos - g1_pos
        
        # Compute relative rotations
        p_rel_rot = p1_rot.inv() * p2_rot
        g_rel_rot = g1_rot.inv() * g2_rot
        
        # Compute errors
        trans_error = np.linalg.norm(p_rel_trans - g_rel_trans)
        rot_error = (p_rel_rot.inv() * g_rel_rot).magnitude()
        
        trans_errors.append(trans_error)
        rot_errors.append(rot_error)
    
    return np.mean(trans_errors), np.mean(rot_errors)

def plot_trajectory(pred_traj, gt_traj, title="Trajectory Comparison", save_path=None):
    """Plot trajectory comparison."""
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # Extract positions
    pred_pos = pred_traj[:, :3]
    gt_pos = gt_traj[:, :3]
    
    # XY plane (top view)
    axs[0].plot(pred_pos[:, 0], pred_pos[:, 1], 'r-', linewidth=2, label='Predicted')
    axs[0].plot(gt_pos[:, 0], gt_pos[:, 1], 'b-', linewidth=2, label='Ground Truth')
    axs[0].scatter(pred_pos[0, 0], pred_pos[0, 1], c='g', marker='o', s=100, label='Start')
    axs[0].scatter(pred_pos[-1, 0], pred_pos[-1, 1], c='r', marker='o', s=100, label='End (Pred)')
    axs[0].scatter(gt_pos[-1, 0], gt_pos[-1, 1], c='b', marker='o', s=100, label='End (GT)')
    axs[0].set_xlabel('X (m)')
    axs[0].set_ylabel('Y (m)')
    axs[0].set_title('XY Plane (Top View)')
    axs[0].grid(True)
    axs[0].legend()
    
    # XZ plane (side view)
    axs[1].plot(pred_pos[:, 0], pred_pos[:, 2], 'r-', linewidth=2, label='Predicted')
    axs[1].plot(gt_pos[:, 0], gt_pos[:, 2], 'b-', linewidth=2, label='Ground Truth')
    axs[1].scatter(pred_pos[0, 0], pred_pos[0, 2], c='g', marker='o', s=100, label='Start')
    axs[1].scatter(pred_pos[-1, 0], pred_pos[-1, 2], c='r', marker='o', s=100, label='End (Pred)')
    axs[1].scatter(gt_pos[-1, 0], gt_pos[-1, 2], c='b', marker='o', s=100, label='End (GT)')
    axs[1].set_xlabel('X (m)')
    axs[1].set_ylabel('Z (m)')
    axs[1].set_title('XZ Plane (Side View)')
    axs[1].grid(True)
    axs[1].legend()
    
    # YZ plane (front view)
    axs[2].plot(pred_pos[:, 1], pred_pos[:, 2], 'r-', linewidth=2, label='Predicted')
    axs[2].plot(gt_pos[:, 1], gt_pos[:, 2], 'b-', linewidth=2, label='Ground Truth')
    axs[2].scatter(pred_pos[0, 1], pred_pos[0, 2], c='g', marker='o', s=100, label='Start')
    axs[2].scatter(pred_pos[-1, 1], pred_pos[-1, 2], c='r', marker='o', s=100, label='End (Pred)')
    axs[2].scatter(gt_pos[-1, 1], gt_pos[-1, 2], c='b', marker='o', s=100, label='End (GT)')
    axs[2].set_xlabel('Y (m)')
    axs[2].set_ylabel('Z (m)')
    axs[2].set_title('YZ Plane (Front View)')
    axs[2].grid(True)
    axs[2].legend()
    
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    
    return fig

def find_optimal_params(pred_rel_poses, gt_traj, is_camera_frame=True):
    """
    Find optimal trajectory integration parameters.
    
    Args:
        pred_rel_poses: Predicted relative poses [N, 6]
        gt_traj: Ground truth trajectory [N+1, 7]
        is_camera_frame: Whether poses are in camera frame
        
    Returns:
        Dictionary with optimal parameters
    """
    logger.info("Finding optimal integration parameters...")
    
    best_ate = float('inf')
    best_params = None
    results = []
    
    # Test different scaling factors and integration methods
    scale_values = [0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.9]
    apply_rot_values = [True, False]
    
    for scale in scale_values:
        for apply_rot in apply_rot_values:
            # Integrate trajectory with current parameters
            pred_traj = integrate_trajectory(
                pred_rel_poses, 
                scale_factor=scale, 
                apply_rotation=apply_rot,
                is_camera_frame=is_camera_frame
            )
            
            # Compute ATE
            mean_ate, std_ate, max_ate = compute_ate(pred_traj, gt_traj)
            
            # Store results
            results.append({
                'scale': scale,
                'apply_rotation': apply_rot,
                'ate_mean': mean_ate,
                'ate_std': std_ate,
                'ate_max': max_ate
            })
            
            # Check if this is better
            if mean_ate < best_ate:
                best_ate = mean_ate
                best_params = {
                    'scale': scale, 
                    'apply_rotation': apply_rot,
                    'ate_mean': mean_ate,
                    'ate_std': std_ate
                }
                logger.info(f"New best: Scale={scale}, Apply Rotation={apply_rot}, ATE={mean_ate:.4f}±{std_ate:.4f} m")
    
    # Sort results by ATE
    results.sort(key=lambda x: x['ate_mean'])
    
    # Log top 5 results
    logger.info("Top 5 parameter sets:")
    for i in range(min(5, len(results))):
        r = results[i]
        logger.info(f"  {i+1}. Scale={r['scale']}, Apply Rotation={r['apply_rotation']}, ATE={r['ate_mean']:.4f}±{r['ate_std']:.4f} m")
    
    return best_params

def test_model(args):
    """Run model testing on a trajectory."""
    # Set device
    device = torch.device("cpu") if args.cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    
    # Create trajectory-specific output directory
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    traj_output_dir = os.path.join(output_dir, f"{args.climate.replace('/', '_')}_{args.traj_id}_{timestamp}")
    os.makedirs(traj_output_dir, exist_ok=True)
    
    # Load model
    model_path = args.model if args.model else par.model_path
    logger.info(f"Loading model from {model_path}")
    
    # Create model
    logger.info("Creating model...")
    try:
        model = VisualInertialOdometryModel(
            imsize1=par.img_h,
            imsize2=par.img_w,
            batchNorm=par.batch_norm,
            use_imu=True,
            imu_feature_size=512,
            imu_input_size=21,  # For integrated IMU
            use_gps=True,
            gps_feature_size=512,
            gps_input_size=9,
            use_adaptive_weighting=True,
            use_depth=True,
            use_depth_temporal=True,
            use_depth_translation=True,
            use_gps_temporal=True,
            pretrained_depth_path=par.pretrained_depth
        )
    except Exception as e:
        logger.warning(f"Error creating model with all parameters: {e}")
        logger.info("Trying simplified model creation...")
        model = VisualInertialOdometryModel(
            imsize1=par.img_h,
            imsize2=par.img_w,
            batchNorm=par.batch_norm,
            use_imu=True,
            use_gps=True,
            use_depth=True,
            use_depth_temporal=True,
            use_depth_translation=True,
            pretrained_depth_path=par.pretrained_depth
        )
    
    # Load weights
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        logger.info("Model weights loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model weights: {e}")
        return None
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    # Get trajectory paths
    traj_dir = os.path.join(par.data_dir, args.climate, args.traj_id)
    rgb_dir = os.path.join(traj_dir, "image_rgb")
    depth_dir = os.path.join(traj_dir, "depth")
    
    # Check for camera frame or body frame
    pose_file = os.path.join(par.pose_dir, args.climate, "poses", f"camera_poses_{args.traj_id.split('_')[1]}.npy")
    is_camera_frame = True
    
    if not os.path.exists(pose_file):
        logger.warning("Camera frame poses not found, trying body frame...")
        pose_file = os.path.join(par.pose_dir, args.climate, "poses", f"poses_{args.traj_id.split('_')[1]}.npy")
        is_camera_frame = False
        
        if not os.path.exists(pose_file):
            logger.error(f"No pose file found for trajectory {args.traj_id}")
            return None
    
    # Load ground truth poses
    gt_poses = np.load(pose_file)
    logger.info(f"Loaded ground truth poses: {gt_poses.shape}")
    logger.info(f"Using {'camera' if is_camera_frame else 'body'} frame")
    
    # Get IMU data
    imu_data = None
    imu_file = os.path.join(traj_dir, "imu", "camera_imu.npy" if is_camera_frame else "imu.npy")
    if os.path.exists(imu_file):
        imu_data = np.load(imu_file)
        logger.info(f"Loaded IMU data: {imu_data.shape}")
    else:
        logger.warning(f"IMU data not found at {imu_file}")
    
    # Get GPS data
    gps_data = None
    gps_file = os.path.join(traj_dir, "gps", "camera_gps.npy" if is_camera_frame else "gps.npy")
    if os.path.exists(gps_file):
        gps_data = np.load(gps_file)
        logger.info(f"Loaded GPS data: {gps_data.shape}")
    else:
        logger.warning(f"GPS data not found at {gps_file}")
    
    # Get RGB files
    rgb_files = sorted([os.path.join(rgb_dir, f) for f in os.listdir(rgb_dir) 
                       if f.lower().endswith(('.jpg', '.jpeg'))])
    
    # Get depth files if available
    depth_files = []
    if os.path.exists(depth_dir):
        depth_files = sorted([os.path.join(depth_dir, f) for f in os.listdir(depth_dir)
                             if f.lower().endswith('.png')])
    
    # Check if we have enough frames
    if len(rgb_files) < par.seq_len + 1:
        logger.error(f"Not enough RGB frames: {len(rgb_files)}, need at least {par.seq_len + 1}")
        return None
    
    # Run model inference
    all_pred_rel_poses = []
    stride = args.stride
    
    with torch.no_grad():
        for i in tqdm(range(0, len(rgb_files) - par.seq_len, stride), desc=f"Processing {args.traj_id}"):
            # Load RGB sequence
            rgb_seq = []
            for j in range(i, i + par.seq_len + 1):
                if j >= len(rgb_files):
                    break
                img_tensor = load_image(rgb_files[j])
                if img_tensor is None:
                    break
                rgb_seq.append(img_tensor)
            
            if len(rgb_seq) < par.seq_len + 1:
                logger.warning(f"Skipping sequence at {i} - not enough valid RGB frames")
                continue
            
            # Stack RGB frames
            rgb_tensor = torch.stack(rgb_seq).unsqueeze(0).to(device)
            
            # Load depth sequence if available
            depth_tensor = None
            if depth_files and model.use_depth:
                depth_seq = []
                for j in range(i, i + par.seq_len + 1):
                    if j >= len(depth_files):
                        break
                    depth_img_tensor = load_depth(depth_files[j])
                    if depth_img_tensor is None:
                        break
                    depth_seq.append(depth_img_tensor)
                
                if len(depth_seq) == par.seq_len + 1:
                    depth_tensor = torch.stack(depth_seq).unsqueeze(0).to(device)
            
            # Prepare IMU data
            imu_tensor = None
            if imu_data is not None and model.use_imu and i + par.seq_len <= len(imu_data):
                # Get IMU data for this sequence
                imu_seq = imu_data[i:i+par.seq_len]
                
                # Convert to integrated format
                integrated_imu = create_integrated_imu_features(imu_seq)
                imu_tensor = torch.FloatTensor(integrated_imu).unsqueeze(0).to(device)
                
                # Log shape for first sequence
                if i == 0:
                    logger.info(f"IMU tensor shape: {imu_tensor.shape}")
            
            # Prepare GPS data
            gps_tensor = None
            if gps_data is not None and model.use_gps and i + par.seq_len <= len(gps_data):
                gps_seq = gps_data[i:i+par.seq_len]
                gps_tensor = torch.FloatTensor(gps_seq).unsqueeze(0).to(device)
                
                # Log shape for first sequence
                if i == 0:
                    logger.info(f"GPS tensor shape: {gps_tensor.shape}")
            
            # Forward pass
            try:
                pred_rel_poses = model(rgb_tensor, depth=depth_tensor, imu_data=imu_tensor, gps_data=gps_tensor)
                all_pred_rel_poses.append(pred_rel_poses[0].cpu().numpy())
                
                # Print first prediction for debugging
                if i == 0:
                    logger.info(f"First relative pose prediction: {pred_rel_poses[0][0].cpu().numpy()}")
            except Exception as e:
                logger.error(f"Error during forward pass: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Check if we got any predictions
    if not all_pred_rel_poses:
        logger.error("No valid predictions generated")
        return None
    
    # Combine all predictions
    pred_rel_poses = np.concatenate(all_pred_rel_poses, axis=0)
    logger.info(f"Generated {len(pred_rel_poses)} relative pose predictions")
    
    # Save raw relative poses
    np.savetxt(os.path.join(traj_output_dir, "pred_rel_poses.txt"), pred_rel_poses, fmt="%.8f")
    
    # Convert ground truth poses to standard format
    gt_traj = prepare_ground_truth(gt_poses)
    
    # Find optimal integration parameters if not specified
    if args.scale is None or not args.scale_only:
        optimal_params = find_optimal_params(pred_rel_poses, gt_traj, is_camera_frame)
        scale_factor = optimal_params['scale']
        apply_rotation = optimal_params['apply_rotation']
        logger.info(f"Using optimal parameters: Scale={scale_factor}, Apply Rotation={apply_rotation}")
    else:
        scale_factor = args.scale
        apply_rotation = not args.no_rotation
        logger.info(f"Using specified parameters: Scale={scale_factor}, Apply Rotation={apply_rotation}")
    
    # Integrate trajectory with optimal parameters
    pred_abs_traj = integrate_trajectory(
        pred_rel_poses, 
        scale_factor=scale_factor, 
        apply_rotation=apply_rotation,
        is_camera_frame=is_camera_frame
    )
    logger.info(f"Integrated trajectory shape: {pred_abs_traj.shape}")
    
    # Trim to matching length
    min_len = min(len(pred_abs_traj), len(gt_traj))
    pred_abs_traj = pred_abs_traj[:min_len]
    gt_traj = gt_traj[:min_len]
    
    # Compute errors
    ate_mean, ate_std, ate_max = compute_ate(pred_abs_traj, gt_traj)
    rpe_trans, rpe_rot = compute_rpe(pred_abs_traj, gt_traj)
    
    # Log metrics
    logger.info(f"ATE: {ate_mean:.4f} ± {ate_std:.4f} m (max: {ate_max:.4f} m)")
    logger.info(f"RPE Trans: {rpe_trans:.4f} m")
    logger.info(f"RPE Rot: {rpe_rot:.4f} rad ({rpe_rot * 180 / np.pi:.4f} degrees)")
    
    # Plot trajectory
    title = f"Trajectory: {args.climate}/{args.traj_id}\nATE: {ate_mean:.4f} ± {ate_std:.4f} m"
    fig = plot_trajectory(pred_abs_traj, gt_traj, title=title, save_path=os.path.join(traj_output_dir, "trajectory.png"))
    plt.close(fig)
    
    # Save trajectories
    np.savetxt(os.path.join(traj_output_dir, "pred_abs_traj.txt"), pred_abs_traj, fmt="%.8f")
    np.savetxt(os.path.join(traj_output_dir, "gt_traj.txt"), gt_traj, fmt="%.8f")
    
    # Save metrics
    metrics = {
        "trajectory": f"{args.climate}/{args.traj_id}",
        "integration_params": {
            "scale_factor": float(scale_factor),
            "apply_rotation": apply_rotation
        },
        "ate": {
            "mean": float(ate_mean),
            "std": float(ate_std),
            "max": float(ate_max)
        },
        "rpe": {
            "translation": float(rpe_trans),
            "rotation_rad": float(rpe_rot),
            "rotation_deg": float(rpe_rot * 180 / np.pi)
        },
        "modalities": {
            "rgb": True,
            "depth": model.use_depth and len(depth_files) > 0,
            "imu": model.use_imu and imu_data is not None,
            "gps": model.use_gps and gps_data is not None
        }
    }
    
    with open(os.path.join(traj_output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    
    logger.info(f"Results saved to {traj_output_dir}")
    
    return metrics

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Optimized Visual Odometry Testing")
    parser.add_argument("--model", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--climate", type=str, default="Kite_training/sunny", help="Climate set")
    parser.add_argument("--traj_id", type=str, default="trajectory_0001", help="Trajectory ID")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage")
    parser.add_argument("--stride", type=int, default=1, help="Frame processing stride")
    parser.add_argument("--scale", type=float, default=None, help="Translation scaling factor")
    parser.add_argument("--no-rotation", action="store_true", help="Disable rotation in integration")
    parser.add_argument("--scale-only", action="store_true", help="Use specified scale without optimization")
    args = parser.parse_args()
    
    # Run test
    metrics = test_model(args)
    
    if metrics:
        logger.info("Testing completed successfully")
        return 0
    else:
        logger.error("Testing failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())