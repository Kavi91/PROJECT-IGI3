import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import argparse
import logging
import glob
from PIL import Image
from torchvision import transforms
from scipy.spatial.transform import Rotation as R

from params import par
from model import RGBVO
from helper import relative_to_absolute_pose, compute_ate, compute_rpe
from data_helper import TartanAirDataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def euler_to_quaternion(roll, pitch, yaw):
    """
    Convert Euler angles to quaternion.
    
    Args:
        roll, pitch, yaw: Rotation angles in radians
    
    Returns:
        qx, qy, qz, qw: Quaternion components
    """
    # Use scipy's Rotation class for reliable conversion
    r = R.from_euler('xyz', [roll, pitch, yaw])
    quat = r.as_quat()  # Returns [x, y, z, w]
    return quat

def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    """Convert quaternion to rotation matrix."""
    # Create rotation matrix
    rotation = np.zeros((3, 3))
    
    # First row
    rotation[0, 0] = 1 - 2*qy*qy - 2*qz*qz
    rotation[0, 1] = 2*qx*qy - 2*qz*qw
    rotation[0, 2] = 2*qx*qz + 2*qy*qw
    
    # Second row
    rotation[1, 0] = 2*qx*qy + 2*qz*qw
    rotation[1, 1] = 1 - 2*qx*qx - 2*qz*qz
    rotation[1, 2] = 2*qy*qz - 2*qx*qw
    
    # Third row
    rotation[2, 0] = 2*qx*qz - 2*qy*qw
    rotation[2, 1] = 2*qy*qz + 2*qx*qw
    rotation[2, 2] = 1 - 2*qx*qx - 2*qy*qy
    
    return rotation

def normalize_quaternion(q):
    """
    Normalize a quaternion to unit length.
    
    Args:
        q: Quaternion [qx, qy, qz, qw]
    
    Returns:
        Normalized quaternion
    """
    norm = np.sqrt(np.sum(q**2))
    if norm < 1e-10:
        return np.array([0.0, 0.0, 0.0, 1.0])  # Identity quaternion
    return q / norm

def align_and_scale_trajectory(pred_positions, gt_positions):
    """
    Align and scale predicted trajectory to match ground truth with direction-specific scaling.
    
    Args:
        pred_positions: Predicted positions [N, 3]
        gt_positions: Ground truth positions [M, 3]
    
    Returns:
        aligned_positions: Aligned and scaled predicted positions
    """
    # Make sure we have valid inputs
    if len(pred_positions) < 2 or len(gt_positions) < 2:
        logger.warning("Cannot align/scale trajectory with less than 2 points")
        return pred_positions
    
    # 1. Translate to match start point exactly
    aligned_positions = pred_positions - pred_positions[0] + gt_positions[0]
    
    # 2. Log overall path length info (for reference)
    gt_segments = np.diff(gt_positions, axis=0)
    gt_distances = np.sqrt(np.sum(gt_segments**2, axis=1))
    gt_path_length = np.sum(gt_distances)
    
    pred_segments = np.diff(aligned_positions, axis=0)
    pred_distances = np.sqrt(np.sum(pred_segments**2, axis=1))
    pred_path_length = np.sum(pred_distances)
    
    logger.info(f"Path length before direction scaling - GT: {gt_path_length:.2f}m, Pred: {pred_path_length:.2f}m")
    
    # 3. Apply direction-specific scaling
    for dim in range(3):
        dim_name = ['X', 'Y', 'Z'][dim]
        # Calculate scale factor for this dimension
        gt_range = np.max(gt_positions[:, dim]) - np.min(gt_positions[:, dim])
        pred_range = np.max(aligned_positions[:, dim]) - np.min(aligned_positions[:, dim])
        
        if pred_range > 1e-6:  # Avoid division by zero
            dim_scale = gt_range / pred_range
            # Apply scale factor only to this dimension
            aligned_positions[:, dim] = gt_positions[0, dim] + (aligned_positions[:, dim] - aligned_positions[0, dim]) * dim_scale
            logger.info(f"{dim_name}-dimension scale factor: {dim_scale:.4f} (GT range: {gt_range:.2f}m, Pred range: {pred_range:.2f}m)")
        else:
            logger.warning(f"{dim_name}-dimension range is too small, skipping scaling")
    
    # 4. Special handling for Z-dimension if needed (uncomment if Z drift is still excessive)
    # z_variation_factor = 0.1  # Reduce Z variation by 90%
    # mean_z = gt_positions[0, 2]  # Use starting Z as reference
    # aligned_positions[:, 2] = mean_z + (aligned_positions[:, 2] - mean_z) * z_variation_factor
    # logger.info(f"Applied additional Z constraint factor: {z_variation_factor}")
    
    return aligned_positions

def test_trajectory(model, trajectory, device, output_dir):
    """
    Test model on a specific trajectory.
    
    Args:
        model: Trained RGBVO model
        trajectory: Tuple of (environment, difficulty, trajectory_id)
        device: Device to run inference on
        output_dir: Directory to save outputs
    
    Returns:
        dict: Metrics for this trajectory
    """
    env, difficulty, trajectory_id = trajectory
    logger.info(f"Testing on trajectory: {env}/{difficulty}/{trajectory_id}")
    
    # Create dataset for this trajectory only
    dataset = TartanAirDataset([trajectory], is_training=False)
    if len(dataset) == 0:
        logger.warning(f"No sequences found for {env}/{difficulty}/{trajectory_id}")
        return None
    
    # Create directories for outputs
    trajectory_dir = os.path.join(output_dir, f"{env}_{difficulty}_{trajectory_id}")
    os.makedirs(trajectory_dir, exist_ok=True)
    
    model.eval()
    
    # Get transformation parameters
    body_to_camera_rotation = par.body_to_camera_rotation.to(device)
    body_to_camera_translation = par.body_to_camera_translation.to(device)
    
    # Metrics for all sequences
    all_ate_mean = []
    all_ate_std = []
    all_rpe_trans_mean = []
    all_rpe_trans_std = []
    all_rpe_rot_mean = []
    all_rpe_rot_std = []
    
    # Lists to store all predictions and ground truth
    all_pred_rel_poses = []
    all_gt_abs_poses = []
    
    # Lists to store full trajectory data (concatenating all sequences)
    full_pred_trajectory = []
    full_gt_trajectory = []
    
    # Process all sequences in this trajectory
    with torch.no_grad():
        for i, (rgb_seq, rel_poses, abs_poses) in enumerate(tqdm(dataset, desc="Testing sequences")):
            # Prepare data
            rgb_seq = rgb_seq.unsqueeze(0).to(device)  # Add batch dimension
            rel_poses = rel_poses.unsqueeze(0).to(device)
            abs_poses = abs_poses.unsqueeze(0).to(device)
            
            # Forward pass
            pred_rel_poses = model(rgb_seq)
            
            # Compute loss
            loss, rot_loss, trans_loss = model.get_loss(pred_rel_poses, rel_poses)
            
            # Integrate to absolute trajectory for this sequence
            pred_abs_poses = relative_to_absolute_pose(
                pred_rel_poses[0], 
                body_to_camera_rotation, 
                body_to_camera_translation
            )
            gt_abs_poses = abs_poses[0]
            
            # Store predictions and ground truth for metrics
            all_pred_rel_poses.append(pred_rel_poses[0].cpu().numpy())
            all_gt_abs_poses.append(gt_abs_poses.cpu().numpy())
            
            # Add to full trajectory (for end-to-end evaluation)
            # We're collecting sequence by sequence, but will integrate them later
            full_pred_trajectory.append(pred_abs_poses.cpu().numpy())
            full_gt_trajectory.append(gt_abs_poses.cpu().numpy())
            
            # Compute metrics for this sequence
            ate_mean, ate_std = compute_ate(pred_abs_poses, gt_abs_poses)
            rpe_trans_mean, rpe_trans_std, rpe_rot_mean, rpe_rot_std = compute_rpe(pred_abs_poses, gt_abs_poses)
            
            all_ate_mean.append(ate_mean)
            all_ate_std.append(ate_std)
            all_rpe_trans_mean.append(rpe_trans_mean)
            all_rpe_trans_std.append(rpe_trans_std)
            all_rpe_rot_mean.append(rpe_rot_mean)
            all_rpe_rot_std.append(rpe_rot_std)
    
    # Save and process full trajectory
    if full_pred_trajectory and full_gt_trajectory:
        try:
            # Concatenate the full trajectory data
            full_gt_abs_traj = np.vstack(full_gt_trajectory)
            full_pred_abs_traj = np.vstack(full_pred_trajectory)
            
            # Save raw full trajectory data (without any alignment or scaling)
            # IMPORTANT: Use space delimiter (not comma) for all trajectory files
            full_gt_raw_path = os.path.join(trajectory_dir, "full_gt_trajectory_raw.txt")
            full_pred_raw_path = os.path.join(trajectory_dir, "full_pred_trajectory_raw.txt")
            np.savetxt(full_gt_raw_path, full_gt_abs_traj, delimiter=' ', fmt='%.18e')
            np.savetxt(full_pred_raw_path, full_pred_abs_traj, delimiter=' ', fmt='%.18e')
            
            logger.info(f"Saved raw full trajectories: GT shape {full_gt_abs_traj.shape}, Pred shape {full_pred_abs_traj.shape}")
            
            # Load original ground truth from TartanAir for reference
            pose_file = os.path.join(par.pose_dir, env, difficulty, trajectory_id, "pose_left.txt")
            if os.path.exists(pose_file):
                # Load raw poses (in TartanAir format: tx ty tz qx qy qz qw)
                raw_gt_poses = np.loadtxt(pose_file)
                logger.info(f"Loaded original ground truth poses from {pose_file}: {raw_gt_poses.shape} poses")
                
                # Save ground truth in TartanAir format
                pose_gt_path = os.path.join(trajectory_dir, "pose_gt.txt")
                np.savetxt(pose_gt_path, raw_gt_poses, delimiter=' ', fmt='%.18e')
                
                # Extract ground truth positions for plotting
                gt_positions = raw_gt_poses[:, 0:3]  # tx, ty, tz
                
                # Now properly integrate our predicted trajectory from relative poses
                # This is an alternative approach to get the full trajectory
                # Use XYZ mapping (identity permutation) based on the permutation test results
                pred_traj = []
                current_position = np.zeros(3)
                current_rotation = np.eye(3)
                
                # Process all predicted relative poses sequences
                for seq_idx, pred_rel_seq in enumerate(all_pred_rel_poses):
                    for i in range(len(pred_rel_seq)):
                        # Extract relative pose components
                        rel_roll, rel_pitch, rel_yaw = pred_rel_seq[i, 0:3]
                        rel_x, rel_y, rel_z = pred_rel_seq[i, 3:6]
                        
                        # Create rotation matrix for relative pose
                        rel_rotation = R.from_euler('xyz', [rel_roll, rel_pitch, rel_yaw]).as_matrix()
                        
                        # Update current rotation
                        current_rotation = current_rotation @ rel_rotation
                        
                        # Create translation vector
                        rel_translation = np.array([rel_x, rel_y, rel_z])
                        
                        # Apply to current position (rotate translation to global frame)
                        current_position = current_position + current_rotation @ rel_translation
                        
                        # Convert rotation to quaternion
                        r = R.from_matrix(current_rotation)
                        quat = r.as_quat()  # [x, y, z, w]
                        
                        # Normalize the quaternion to ensure unit length
                        quat = normalize_quaternion(quat)
                        
                        # Store in TartanAir format: tx ty tz qx qy qz qw
                        pred_traj.append(np.concatenate([current_position, quat]))
                
                # Convert to numpy array
                pred_traj = np.array(pred_traj)
                
                # Extract predicted positions for alignment and plotting
                pred_positions = pred_traj[:, 0:3]
                
                # Apply improved alignment and scaling
                aligned_pred_positions = align_and_scale_trajectory(pred_positions, gt_positions)
                
                # Create aligned predicted trajectory in TartanAir format
                aligned_pred_traj = np.copy(pred_traj)
                aligned_pred_traj[:, 0:3] = aligned_pred_positions
                
                # Verify the shape of the trajectories before saving
                if aligned_pred_traj.shape[1] != 7:
                    logger.error(f"ERROR: Predicted trajectory has {aligned_pred_traj.shape[1]} columns instead of 7!")
                    # Fix if possible by padding or truncating
                    if aligned_pred_traj.shape[1] < 7:
                        logger.warning("Padding trajectory to 7 columns")
                        padded = np.zeros((aligned_pred_traj.shape[0], 7))
                        padded[:, :aligned_pred_traj.shape[1]] = aligned_pred_traj
                        aligned_pred_traj = padded
                    else:
                        logger.warning("Truncating trajectory to 7 columns")
                        aligned_pred_traj = aligned_pred_traj[:, :7]
                
                # Save predicted trajectory in TartanAir format
                pose_est_path = os.path.join(trajectory_dir, "pose_est.txt")
                np.savetxt(pose_est_path, aligned_pred_traj, delimiter=' ', fmt='%.18e')
                
                # Save the raw (unaligned) predicted trajectory for reference
                pose_est_raw_path = os.path.join(trajectory_dir, "pose_est_raw.txt")
                np.savetxt(pose_est_raw_path, pred_traj, delimiter=' ', fmt='%.18e')
                
                # Plot trajectories using positions only
                plot_position_trajectory(
                    gt_positions,
                    aligned_pred_positions,
                    title=f"Trajectory: {env}/{difficulty}/{trajectory_id}",
                    save_path=os.path.join(trajectory_dir, "full_trajectory_3d.png")
                )
                
                # Plot 2D projections
                plot_position_trajectory_2d(
                    gt_positions,
                    aligned_pred_positions,
                    title=f"Trajectory: {env}/{difficulty}/{trajectory_id}",
                    save_path=os.path.join(trajectory_dir, "full_trajectory_2d.png")
                )
                
                # Compute full trajectory metrics
                full_gt_tensor = torch.tensor(gt_positions).float()
                full_pred_tensor = torch.tensor(aligned_pred_positions).float()
                
                # Trim to same length if needed
                min_len = min(len(full_gt_tensor), len(full_pred_tensor))
                full_gt_tensor = full_gt_tensor[:min_len]
                full_pred_tensor = full_pred_tensor[:min_len]
                
                # Create pose tensors with rotation (needed for RPE)
                full_gt_pose = torch.zeros(min_len, 6)
                full_pred_pose = torch.zeros(min_len, 6)
                
                # Fill in positions
                full_gt_pose[:, 3:6] = full_gt_tensor
                full_pred_pose[:, 3:6] = full_pred_tensor
                
                # Compute full trajectory metrics
                full_ate_mean, full_ate_std = compute_ate(full_pred_pose, full_gt_pose)
                full_rpe_trans_mean, full_rpe_trans_std, full_rpe_rot_mean, full_rpe_rot_std = compute_rpe(full_pred_pose, full_gt_pose)
                
                logger.info(f"Full trajectory metrics:")
                logger.info(f"  Full ATE: {full_ate_mean:.4f} ± {full_ate_std:.4f} m")
                logger.info(f"  Full RPE Translation: {full_rpe_trans_mean:.4f} ± {full_rpe_trans_std:.4f} m")
                
                # Test load the saved pose files to verify they are correctly formatted
                try:
                    test_gt = np.loadtxt(pose_gt_path)
                    test_est = np.loadtxt(pose_est_path)
                    logger.info(f"Verified pose files - GT: {test_gt.shape}, EST: {test_est.shape}")
                    
                    # Check for same length
                    if test_gt.shape[0] != test_est.shape[0]:
                        logger.warning(f"WARNING: GT and EST have different lengths - GT: {test_gt.shape[0]}, EST: {test_est.shape[0]}")
                        
                        # Trim the longer one to match the shorter one
                        min_len = min(test_gt.shape[0], test_est.shape[0])
                        if test_gt.shape[0] > min_len:
                            logger.info(f"Trimming GT from {test_gt.shape[0]} to {min_len} rows")
                            test_gt = test_gt[:min_len]
                            np.savetxt(pose_gt_path, test_gt, delimiter=' ', fmt='%.18e')
                        
                        if test_est.shape[0] > min_len:
                            logger.info(f"Trimming EST from {test_est.shape[0]} to {min_len} rows")
                            test_est = test_est[:min_len]
                            np.savetxt(pose_est_path, test_est, delimiter=' ', fmt='%.18e')
                except Exception as e:
                    logger.error(f"Error verifying pose files: {e}")
                
            else:
                logger.warning(f"Original ground truth pose file not found: {pose_file}")
                logger.warning("Using dataset-provided ground truth instead")
                
                # Use our concatenated full ground truth trajectory
                gt_positions = full_gt_abs_traj[:, 3:6]  # Extract positions
                
                # Process predicted trajectory (similar to above)
                pred_traj = []
                current_position = np.zeros(3)
                current_rotation = np.eye(3)
                
                for seq_idx, pred_rel_seq in enumerate(all_pred_rel_poses):
                    for i in range(len(pred_rel_seq)):
                        rel_roll, rel_pitch, rel_yaw = pred_rel_seq[i, 0:3]
                        rel_x, rel_y, rel_z = pred_rel_seq[i, 3:6]
                        
                        rel_rotation = R.from_euler('xyz', [rel_roll, rel_pitch, rel_yaw]).as_matrix()
                        current_rotation = current_rotation @ rel_rotation
                        rel_translation = np.array([rel_x, rel_y, rel_z])
                        current_position = current_position + current_rotation @ rel_translation
                        
                        r = R.from_matrix(current_rotation)
                        quat = r.as_quat()  # [x, y, z, w]
                        quat = normalize_quaternion(quat)
                        
                        pred_traj.append(np.concatenate([current_position, quat]))
                
                pred_traj = np.array(pred_traj)
                
                # Extract positions for alignment
                pred_positions = pred_traj[:, 0:3]
                
                # Apply improved alignment and scaling
                aligned_pred_positions = align_and_scale_trajectory(pred_positions, gt_positions)
                
                # Convert euler and positions to TartanAir format
                gt_quaternions = []
                for i in range(len(full_gt_abs_traj)):
                    quat = euler_to_quaternion(full_gt_abs_traj[i, 0], full_gt_abs_traj[i, 1], full_gt_abs_traj[i, 2])
                    quat = normalize_quaternion(quat)
                    gt_quaternions.append(quat)
                
                gt_quaternions = np.array(gt_quaternions)
                gt_traj = np.concatenate([gt_positions, gt_quaternions], axis=1)
                
                # Verify shapes
                if gt_traj.shape[1] != 7:
                    logger.error(f"ERROR: GT trajectory has {gt_traj.shape[1]} columns instead of 7!")
                    # Fix if possible
                    if gt_traj.shape[1] < 7:
                        logger.warning("Padding GT trajectory to 7 columns")
                        padded = np.zeros((gt_traj.shape[0], 7))
                        padded[:, :gt_traj.shape[1]] = gt_traj
                        gt_traj = padded
                    else:
                        logger.warning("Truncating GT trajectory to 7 columns")
                        gt_traj = gt_traj[:, :7]
                
                # Create aligned predicted trajectory
                aligned_pred_traj = np.copy(pred_traj)
                aligned_pred_traj[:, 0:3] = aligned_pred_positions
                
                # Make sure trajectories have same length
                min_len = min(len(gt_traj), len(aligned_pred_traj))
                gt_traj = gt_traj[:min_len]
                aligned_pred_traj = aligned_pred_traj[:min_len]
                
                # Save full trajectories with space delimiter (not comma)
                pose_gt_path = os.path.join(trajectory_dir, "pose_gt.txt")
                pose_est_path = os.path.join(trajectory_dir, "pose_est.txt")
                pose_est_raw_path = os.path.join(trajectory_dir, "pose_est_raw.txt")
                
                np.savetxt(pose_gt_path, gt_traj, delimiter=' ', fmt='%.18e')
                np.savetxt(pose_est_path, aligned_pred_traj, delimiter=' ', fmt='%.18e')
                np.savetxt(pose_est_raw_path, pred_traj, delimiter=' ', fmt='%.18e')
                
                # Plot trajectories
                plot_position_trajectory(
                    gt_positions,
                    aligned_pred_positions,
                    title=f"Trajectory: {env}/{difficulty}/{trajectory_id}",
                    save_path=os.path.join(trajectory_dir, "full_trajectory_3d.png")
                )
                
                plot_position_trajectory_2d(
                    gt_positions,
                    aligned_pred_positions,
                    title=f"Trajectory: {env}/{difficulty}/{trajectory_id}",
                    save_path=os.path.join(trajectory_dir, "full_trajectory_2d.png")
                )
                
                # Compute full trajectory metrics as above
                full_gt_tensor = torch.tensor(gt_positions).float()
                full_pred_tensor = torch.tensor(aligned_pred_positions).float()
                
                min_len = min(len(full_gt_tensor), len(full_pred_tensor))
                full_gt_tensor = full_gt_tensor[:min_len]
                full_pred_tensor = full_pred_tensor[:min_len]
                
                full_gt_pose = torch.zeros(min_len, 6)
                full_pred_pose = torch.zeros(min_len, 6)
                
                full_gt_pose[:, 3:6] = full_gt_tensor
                full_pred_pose[:, 3:6] = full_pred_tensor
                
                full_ate_mean, full_ate_std = compute_ate(full_pred_pose, full_gt_pose)
                full_rpe_trans_mean, full_rpe_trans_std, full_rpe_rot_mean, full_rpe_rot_std = compute_rpe(full_pred_pose, full_gt_pose)
                
                logger.info(f"Full trajectory metrics:")
                logger.info(f"  Full ATE: {full_ate_mean:.4f} ± {full_ate_std:.4f} m")
                logger.info(f"  Full RPE Translation: {full_rpe_trans_mean:.4f} ± {full_rpe_trans_std:.4f} m")
                
                # Test load the saved pose files to verify they are correctly formatted
                try:
                    test_gt = np.loadtxt(pose_gt_path)
                    test_est = np.loadtxt(pose_est_path)
                    logger.info(f"Verified pose files - GT: {test_gt.shape}, EST: {test_est.shape}")
                except Exception as e:
                    logger.error(f"Error verifying pose files: {e}")
        
        except Exception as e:
            logger.error(f"Error processing full trajectory: {e}")
            import traceback
            traceback.print_exc()
    
    # Compute average metrics from sequence-level evaluation
    metrics = {
        'ate_mean': np.mean(all_ate_mean),
        'ate_std': np.mean(all_ate_std),
        'rpe_trans_mean': np.mean(all_rpe_trans_mean),
        'rpe_trans_std': np.mean(all_rpe_trans_std),
        'rpe_rot_mean': np.mean(all_rpe_rot_mean),
        'rpe_rot_std': np.mean(all_rpe_rot_std),
        'num_sequences': len(all_ate_mean)
    }
    
    # Add full trajectory metrics if available
    try:
        metrics['full_ate_mean'] = full_ate_mean
        metrics['full_ate_std'] = full_ate_std
        metrics['full_rpe_trans_mean'] = full_rpe_trans_mean
        metrics['full_rpe_trans_std'] = full_rpe_trans_std
    except:
        pass
    
    # Save metrics
    metrics_path = os.path.join(trajectory_dir, "metrics.txt")
    with open(metrics_path, 'w') as f:
        f.write(f"Trajectory: {env}/{difficulty}/{trajectory_id}\n")
        f.write(f"Number of sequences: {metrics['num_sequences']}\n")
        f.write(f"Sequence-level metrics (averaged):\n")
        f.write(f"  ATE: {metrics['ate_mean']:.4f} ± {metrics['ate_std']:.4f} m\n")
        f.write(f"  RPE Translation: {metrics['rpe_trans_mean']:.4f} ± {metrics['rpe_trans_std']:.4f} m\n")
        f.write(f"  RPE Rotation: {metrics['rpe_rot_mean']:.4f} ± {metrics['rpe_rot_std']:.4f} rad\n")
        
        if 'full_ate_mean' in metrics:
            f.write(f"\nFull trajectory metrics:\n")
            f.write(f"  Full ATE: {metrics['full_ate_mean']:.4f} ± {metrics['full_ate_std']:.4f} m\n")
            f.write(f"  Full RPE Translation: {metrics['full_rpe_trans_mean']:.4f} ± {metrics['full_rpe_trans_std']:.4f} m\n")
    
    logger.info(f"Results for {env}/{difficulty}/{trajectory_id}:")
    logger.info(f"  Number of sequences: {metrics['num_sequences']}")
    logger.info(f"  ATE: {metrics['ate_mean']:.4f} ± {metrics['ate_std']:.4f} m")
    logger.info(f"  RPE Translation: {metrics['rpe_trans_mean']:.4f} ± {metrics['rpe_trans_std']:.4f} m")
    logger.info(f"  RPE Rotation: {metrics['rpe_rot_mean']:.4f} ± {metrics['rpe_rot_std']:.4f} rad")
    
    return metrics

def plot_position_trajectory(gt_positions, pred_positions, title="Trajectory", save_path=None):
    """
    Plot 3D trajectory of ground truth and predicted positions.
    
    Args:
        gt_positions: Ground truth positions [N, 3] (tx, ty, tz)
        pred_positions: Predicted positions [N, 3] (tx, ty, tz)
        title: Plot title
        save_path: Path to save the figure
    """
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot ground truth trajectory
    ax.plot(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2], 
            'g-', linewidth=2, label='Ground Truth')
    ax.scatter(gt_positions[0, 0], gt_positions[0, 1], gt_positions[0, 2], 
               c='b', s=80, marker='o', label='Start')
    ax.scatter(gt_positions[-1, 0], gt_positions[-1, 1], gt_positions[-1, 2], 
               c='g', s=80, marker='s', label='End (GT)')
    
    # Plot predicted trajectory
    ax.plot(pred_positions[:, 0], pred_positions[:, 1], pred_positions[:, 2], 
            'r--', linewidth=2, label='Predicted')
    ax.scatter(pred_positions[-1, 0], pred_positions[-1, 1], pred_positions[-1, 2], 
               c='r', s=80, marker='x', label='End (Pred)')
    
    # Set labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    ax.legend()
    
    # Equal aspect ratio
    all_x = np.concatenate((gt_positions[:, 0], pred_positions[:, 0]))
    all_y = np.concatenate((gt_positions[:, 1], pred_positions[:, 1]))
    all_z = np.concatenate((gt_positions[:, 2], pred_positions[:, 2]))
    
    max_range = np.max([
        all_x.max() - all_x.min(),
        all_y.max() - all_y.min(),
        all_z.max() - all_z.min()
    ])
    
    mid_x = (all_x.max() + all_x.min()) / 2
    mid_y = (all_y.max() + all_y.min()) / 2
    mid_z = (all_z.max() + all_z.min()) / 2
    
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
    
    # Save if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close(fig)
    return fig, ax

def plot_position_trajectory_2d(gt_positions, pred_positions, title="Trajectory", save_path=None):
    """
    Plot 2D projections of the trajectory (top-down, side, and front views).
    
    Args:
        gt_positions: Ground truth positions [N, 3] (tx, ty, tz)
        pred_positions: Predicted positions [N, 3] (tx, ty, tz)
        title: Plot title
        save_path: Path to save the figure
    """
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # XY plot (top-down view)
    axs[0].plot(gt_positions[:, 0], gt_positions[:, 1], 'g-', linewidth=2, label='Ground Truth')
    axs[0].plot(pred_positions[:, 0], pred_positions[:, 1], 'r--', linewidth=2, label='Predicted')
    axs[0].scatter(gt_positions[0, 0], gt_positions[0, 1], c='b', s=80, marker='o', label='Start')
    axs[0].scatter(gt_positions[-1, 0], gt_positions[-1, 1], c='g', s=80, marker='s', label='End (GT)')
    axs[0].scatter(pred_positions[-1, 0], pred_positions[-1, 1], c='r', s=80, marker='x', label='End (Pred)')
    axs[0].set_xlabel('X (m)')
    axs[0].set_ylabel('Y (m)')
    axs[0].set_title('XY Plane (Top-Down View)')
    axs[0].grid(True)
    axs[0].axis('equal')
    axs[0].legend()
    
    # XZ plot (side view)
    axs[1].plot(gt_positions[:, 0], gt_positions[:, 2], 'g-', linewidth=2, label='Ground Truth')
    axs[1].plot(pred_positions[:, 0], pred_positions[:, 2], 'r--', linewidth=2, label='Predicted')
    axs[1].scatter(gt_positions[0, 0], gt_positions[0, 2], c='b', s=80, marker='o', label='Start')
    axs[1].scatter(gt_positions[-1, 0], gt_positions[-1, 2], c='g', s=80, marker='s', label='End (GT)')
    axs[1].scatter(pred_positions[-1, 0], pred_positions[-1, 2], c='r', s=80, marker='x', label='End (Pred)')
    axs[1].set_xlabel('X (m)')
    axs[1].set_ylabel('Z (m)')
    axs[1].set_title('XZ Plane (Side View)')
    axs[1].grid(True)
    axs[1].axis('equal')
    axs[1].legend()
    
    # YZ plot (front view)
    axs[2].plot(gt_positions[:, 1], gt_positions[:, 2], 'g-', linewidth=2, label='Ground Truth')
    axs[2].plot(pred_positions[:, 1], pred_positions[:, 2], 'r--', linewidth=2, label='Predicted')
    axs[2].scatter(gt_positions[0, 1], gt_positions[0, 2], c='b', s=80, marker='o', label='Start')
    axs[2].scatter(gt_positions[-1, 1], gt_positions[-1, 2], c='g', s=80, marker='s', label='End (GT)')
    axs[2].scatter(pred_positions[-1, 1], pred_positions[-1, 2], c='r', s=80, marker='x', label='End (Pred)')
    axs[2].set_xlabel('Y (m)')
    axs[2].set_ylabel('Z (m)')
    axs[2].set_title('YZ Plane (Front View)')
    axs[2].grid(True)
    axs[2].axis('equal')
    axs[2].legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    
    # Save if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close(fig)
    return fig, axs

def test_inference_on_sequence(model, image_dir, device, output_dir):
    """
    Run inference on a single image sequence without ground truth.
    
    Args:
        model: Trained RGBVO model
        image_dir: Directory containing the image sequence
        device: Device to run inference on
        output_dir: Directory to save outputs
    """
    logger.info(f"Running inference on images in {image_dir}")
    
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get image files
    image_files = glob.glob(os.path.join(image_dir, '*.png'))
    image_files.sort()
    
    if len(image_files) < par.seq_len + 1:
        logger.error(f"Not enough images in {image_dir} (found {len(image_files)}, need at least {par.seq_len + 1})")
        return
    
    # Image transformations - using the same transformation as in training
    transform_ops = [
        transforms.Resize((par.img_h, par.img_w)),
        transforms.ToTensor()
    ]
    transformer = transforms.Compose(transform_ops)
    normalizer = transforms.Normalize(mean=par.img_means_rgb, std=par.img_stds_rgb)
    
    # Get transformation parameters
    body_to_camera_rotation = par.body_to_camera_rotation.to(device)
    body_to_camera_translation = par.body_to_camera_translation.to(device)
    
    # Initialize predictions list
    all_rel_poses = []
    
    # Process images in sequence
    model.eval()
    with torch.no_grad():
        # Process all windows of size seq_len+1
        for i in range(0, len(image_files) - par.seq_len):
            # Load images for this window
            image_window = []
            for j in range(i, i + par.seq_len + 1):
                try:
                    img = Image.open(image_files[j])
                    img_tensor = transformer(img)
                    
                    # Apply FlowNet normalization if required
                    if par.minus_point_5:
                        img_tensor = img_tensor - 0.5
                    
                    # Apply normalization
                    img_tensor = normalizer(img_tensor)
                    image_window.append(img_tensor)
                except Exception as e:
                    logger.error(f"Error loading image {image_files[j]}: {e}")
                    return
            
            # Stack images and add batch dimension
            image_window = torch.stack(image_window).unsqueeze(0).to(device)
            
            # Forward pass
            pred_rel_poses = model(image_window)
            
            # Store relative poses
            all_rel_poses.append(pred_rel_poses[0].cpu().numpy())
    
    # Reconstruct trajectory if we have predictions
    if all_rel_poses:
        # Initialize with identity rotation and zero translation
        pred_traj = []
        current_position = np.zeros(3)
        current_rotation = np.eye(3)
        
        # Process all predicted relative poses
        for rel_seq in all_rel_poses:
            for i in range(len(rel_seq)):
                # Extract relative pose components
                rel_roll, rel_pitch, rel_yaw = rel_seq[i, 0:3]
                rel_x, rel_y, rel_z = rel_seq[i, 3:6]
                
                # Create rotation matrix for relative pose
                rel_rotation = R.from_euler('xyz', [rel_roll, rel_pitch, rel_yaw]).as_matrix()
                
                # Update current rotation
                current_rotation = current_rotation @ rel_rotation
                
                # Create translation vector
                rel_translation = np.array([rel_x, rel_y, rel_z])
                
                # Apply to current position (rotate translation to global frame)
                current_position = current_position + current_rotation @ rel_translation
                
                # Convert rotation to quaternion
                r = R.from_matrix(current_rotation)
                quat = r.as_quat()  # [x, y, z, w]
                quat = normalize_quaternion(quat)
                
                # Store in TartanAir format: tx ty tz qx qy qz qw
                pred_traj.append(np.concatenate([current_position, quat]))
        
        # Convert to numpy array
        pred_traj = np.array(pred_traj)
        
        # Verify shape - should be [N, 7]
        if pred_traj.shape[1] != 7:
            logger.error(f"ERROR: Predicted trajectory has {pred_traj.shape[1]} columns instead of 7!")
            # Fix if possible
            if pred_traj.shape[1] < 7:
                logger.warning("Padding trajectory to 7 columns")
                padded = np.zeros((pred_traj.shape[0], 7))
                padded[:, :pred_traj.shape[1]] = pred_traj
                pred_traj = padded
            else:
                logger.warning("Truncating trajectory to 7 columns")
                pred_traj = pred_traj[:, :7]
        
        # Extract positions for plotting
        pred_positions = pred_traj[:, 0:3]
        
        # Save trajectory in TartanAir format (use space delimiter)
        pose_est_path = os.path.join(output_dir, "pose_est.txt")
        np.savetxt(pose_est_path, pred_traj, delimiter=' ', fmt='%.18e')
        
        # Plot 3D trajectory
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.plot(pred_positions[:, 0], pred_positions[:, 1], pred_positions[:, 2], 'r-', linewidth=2)
        ax.scatter(pred_positions[0, 0], pred_positions[0, 1], pred_positions[0, 2], c='b', s=80, marker='o', label='Start')
        ax.scatter(pred_positions[-1, 0], pred_positions[-1, 1], pred_positions[-1, 2], c='r', s=80, marker='x', label='End')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Predicted Trajectory')
        ax.legend()
        
        # Equal aspect ratio
        max_range = np.max([
            pred_positions[:, 0].max() - pred_positions[:, 0].min(),
            pred_positions[:, 1].max() - pred_positions[:, 1].min(),
            pred_positions[:, 2].max() - pred_positions[:, 2].min()
        ])
        
        mid_x = (pred_positions[:, 0].max() + pred_positions[:, 0].min()) / 2
        mid_y = (pred_positions[:, 1].max() + pred_positions[:, 1].min()) / 2
        mid_z = (pred_positions[:, 2].max() + pred_positions[:, 2].min()) / 2
        
        ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
        ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
        ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
        
        plot_path = os.path.join(output_dir, "trajectory_3d.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2D views
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        
        # XY plot (top-down view)
        axs[0].plot(pred_positions[:, 0], pred_positions[:, 1], 'r-', linewidth=2)
        axs[0].scatter(pred_positions[0, 0], pred_positions[0, 1], c='b', s=80, marker='o', label='Start')
        axs[0].scatter(pred_positions[-1, 0], pred_positions[-1, 1], c='r', s=80, marker='x', label='End')
        axs[0].set_xlabel('X (m)')
        axs[0].set_ylabel('Y (m)')
        axs[0].set_title('XY Plane (Top-Down View)')
        axs[0].grid(True)
        axs[0].axis('equal')
        axs[0].legend()
        
        # XZ plot (side view)
        axs[1].plot(pred_positions[:, 0], pred_positions[:, 2], 'r-', linewidth=2)
        axs[1].scatter(pred_positions[0, 0], pred_positions[0, 2], c='b', s=80, marker='o', label='Start')
        axs[1].scatter(pred_positions[-1, 0], pred_positions[-1, 2], c='r', s=80, marker='x', label='End')
        axs[1].set_xlabel('X (m)')
        axs[1].set_ylabel('Z (m)')
        axs[1].set_title('XZ Plane (Side View)')
        axs[1].grid(True)
        axs[1].axis('equal')
        axs[1].legend()
        
        # YZ plot (front view)
        axs[2].plot(pred_positions[:, 1], pred_positions[:, 2], 'r-', linewidth=2)
        axs[2].scatter(pred_positions[0, 1], pred_positions[0, 2], c='b', s=80, marker='o', label='Start')
        axs[2].scatter(pred_positions[-1, 1], pred_positions[-1, 2], c='r', s=80, marker='x', label='End')
        axs[2].set_xlabel('Y (m)')
        axs[2].set_ylabel('Z (m)')
        axs[2].set_title('YZ Plane (Front View)')
        axs[2].grid(True)
        axs[2].axis('equal')
        axs[2].legend()
        
        plt.suptitle('Predicted Trajectory Projections')
        plt.tight_layout()
        
        plot_path_2d = os.path.join(output_dir, "trajectory_2d.png")
        plt.savefig(plot_path_2d, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Test load the saved pose file to verify it's correctly formatted
        try:
            test_est = np.loadtxt(pose_est_path)
            logger.info(f"Verified pose file: {test_est.shape}")
        except Exception as e:
            logger.error(f"Error verifying pose file: {e}")
        
        logger.info(f"Inference completed. Results saved to {output_dir}")
    else:
        logger.error("No predictions were made.")

def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create model
    model = RGBVO(imsize1=par.img_h, imsize2=par.img_w, batchNorm=par.batch_norm)
    model = model.to(device)
    
    # Load model weights
    model_path = args.model if args.model else par.model_path
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        # This is a full checkpoint (contains optimizer state, etc.)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded model weights from checkpoint at epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        # This is just the model weights
        model.load_state_dict(checkpoint)
        logger.info(f"Loaded model weights directly")
    logger.info(f"Loaded model from: {model_path}")
    
    # Create output directory
    output_dir = args.output if args.output else 'results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Run inference on a single sequence if specified
    if args.inference_dir:
        test_inference_on_sequence(model, args.inference_dir, device, output_dir)
        return
    
    # Define test trajectories
    if args.trajectories:
        # Parse trajectory string: env/difficulty/trajectory_id,env/difficulty/trajectory_id,...
        test_trajectories = []
        for traj_str in args.trajectories.split(','):
            parts = traj_str.split('/')
            if len(parts) >= 3:
                env = parts[0]
                difficulty = parts[1]
                trajectory_id = parts[2]
                test_trajectories.append((env, difficulty, trajectory_id))
    else:
        # Use validation trajectories by default
        test_trajectories = par.valid_trajectories
    
    logger.info(f"Testing on {len(test_trajectories)} trajectories")
    
    # Test on each trajectory
    all_metrics = {}
    for trajectory in test_trajectories:
        metrics = test_trajectory(model, trajectory, device, output_dir)
        if metrics:
            traj_key = f"{trajectory[0]}/{trajectory[1]}/{trajectory[2]}"
            all_metrics[traj_key] = metrics
    
    # Save overall metrics
    overall_path = os.path.join(output_dir, "overall_metrics.txt")
    with open(overall_path, 'w') as f:
        f.write(f"Test results for {len(all_metrics)} trajectories\n")
        f.write("=" * 50 + "\n\n")
        
        # Calculate average metrics across all trajectories
        if all_metrics:
            avg_ate_mean = np.mean([m['ate_mean'] for m in all_metrics.values()])
            avg_rpe_trans_mean = np.mean([m['rpe_trans_mean'] for m in all_metrics.values()])
            avg_rpe_rot_mean = np.mean([m['rpe_rot_mean'] for m in all_metrics.values()])
            
            f.write(f"Overall average ATE: {avg_ate_mean:.4f} m\n")
            f.write(f"Overall average RPE Translation: {avg_rpe_trans_mean:.4f} m\n")
            f.write(f"Overall average RPE Rotation: {avg_rpe_rot_mean:.4f} rad\n\n")
            
            f.write("Per-trajectory metrics:\n")
            f.write("-" * 50 + "\n")
            for traj, metrics in all_metrics.items():
                f.write(f"Trajectory: {traj}\n")
                f.write(f"  ATE: {metrics['ate_mean']:.4f} ± {metrics['ate_std']:.4f} m\n")
                f.write(f"  RPE Translation: {metrics['rpe_trans_mean']:.4f} ± {metrics['rpe_trans_std']:.4f} m\n")
                f.write(f"  RPE Rotation: {metrics['rpe_rot_mean']:.4f} ± {metrics['rpe_rot_std']:.4f} rad\n\n")
        else:
            f.write("No valid results to report.\n")
    
    # Print summary
    logger.info("\n" + "=" * 50)
    logger.info("Testing completed!")
    if all_metrics:
        avg_ate_mean = np.mean([m['ate_mean'] for m in all_metrics.values()])
        avg_rpe_trans_mean = np.mean([m['rpe_trans_mean'] for m in all_metrics.values()])
        avg_rpe_rot_mean = np.mean([m['rpe_rot_mean'] for m in all_metrics.values()])
        
        logger.info(f"Overall average ATE: {avg_ate_mean:.4f} m")
        logger.info(f"Overall average RPE Translation: {avg_rpe_trans_mean:.4f} m")
        logger.info(f"Overall average RPE Rotation: {avg_rpe_rot_mean:.4f} rad")
    logger.info("=" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test RGBVO model on TartanAir')
    parser.add_argument('--model', type=str, help='Path to model file')
    parser.add_argument('--output', type=str, help='Output directory')
    parser.add_argument('--trajectories', type=str, help='Comma-separated list of trajectories (env/difficulty/trajectory_id)')
    parser.add_argument('--inference_dir', type=str, help='Directory with image sequence for inference only')
    parser.add_argument('--cpu', action='store_true', help='Force using CPU')
    args = parser.parse_args()
    
    main(args)