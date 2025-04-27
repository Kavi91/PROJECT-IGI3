#!/usr/bin/env python3
"""
Debugging script for trajectory integration issues
This script focuses on properly integrating relative poses from a model

python debug_trajectory.py --pred_rel_poses results/Kite_training_sunny_trajectory_0001_20250507-000300/pred_rel_poses.txt --gt_poses /media/krkavinda/New\ Volume/Mid-Air-Dataset/MidAir_processed/Kite_training/sunny/poses/camera_poses_0001.npy --output debug_results


"""

import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.spatial.transform import Rotation
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import parameters
from params import par

def simple_trajectory_integration(rel_poses):
    """
    A very simple integration approach that just accumulates translations
    without using rotations, to check if that's where the issue is.
    
    Args:
        rel_poses: Array of relative poses [N, 6] with [roll, pitch, yaw, x, y, z]
        
    Returns:
        positions: Array of accumulated positions [N+1, 3]
    """
    # Start with zero position
    positions = np.zeros((len(rel_poses) + 1, 3))
    
    # Just accumulate translations directly
    for i in range(len(rel_poses)):
        # Get relative translation
        rel_trans = rel_poses[i, 3:6]
        
        # Add to previous position (no rotation involved)
        positions[i+1] = positions[i] + rel_trans
    
    return positions

def trajectory_integration_zyx(rel_poses):
    """
    Integrate relative poses using ZYX Euler angle convention.
    
    Args:
        rel_poses: Array of relative poses [N, 6] with [roll, pitch, yaw, x, y, z]
        
    Returns:
        abs_poses: Array of absolute poses [N+1, 7] with [x, y, z, qx, qy, qz, qw]
    """
    # First pose is identity
    abs_poses = np.zeros((len(rel_poses) + 1, 7))
    abs_poses[0, 6] = 1.0  # Identity quaternion [0, 0, 0, 1]
    
    # Current state
    curr_pos = np.zeros(3)
    curr_rot = Rotation.identity()
    
    logger.info(f"First few relative poses:")
    for i in range(min(5, len(rel_poses))):
        logger.info(f"  {i}: {rel_poses[i]}")
    
    for i in range(len(rel_poses)):
        # Get relative pose
        rel_euler = rel_poses[i, :3]  # roll, pitch, yaw
        rel_trans = rel_poses[i, 3:6]  # x, y, z
        
        # Convert to rotation object using ZYX convention (yaw, pitch, roll)
        rel_rot = Rotation.from_euler('zyx', rel_euler)
        
        # Update current rotation: curr_rot = curr_rot * rel_rot
        curr_rot = curr_rot * rel_rot
        
        # Transform translation to global frame and add: curr_pos += R * t
        curr_pos = curr_pos + curr_rot.apply(rel_trans)
        
        # Store results
        abs_poses[i+1, :3] = curr_pos
        abs_poses[i+1, 3:7] = curr_rot.as_quat()
    
    return abs_poses

def trajectory_integration_xyz(rel_poses):
    """
    Integrate relative poses using XYZ Euler angle convention.
    
    Args:
        rel_poses: Array of relative poses [N, 6] with [roll, pitch, yaw, x, y, z]
        
    Returns:
        abs_poses: Array of absolute poses [N+1, 7] with [x, y, z, qx, qy, qz, qw]
    """
    # First pose is identity
    abs_poses = np.zeros((len(rel_poses) + 1, 7))
    abs_poses[0, 6] = 1.0  # Identity quaternion [0, 0, 0, 1]
    
    # Current state
    curr_pos = np.zeros(3)
    curr_rot = Rotation.identity()
    
    for i in range(len(rel_poses)):
        # Get relative pose
        rel_euler = rel_poses[i, :3]  # roll, pitch, yaw
        rel_trans = rel_poses[i, 3:6]  # x, y, z
        
        # Convert to rotation object using XYZ convention (roll, pitch, yaw)
        rel_rot = Rotation.from_euler('xyz', rel_euler)
        
        # Update current rotation: curr_rot = curr_rot * rel_rot
        curr_rot = curr_rot * rel_rot
        
        # Transform translation to global frame and add: curr_pos += R * t
        curr_pos = curr_pos + curr_rot.apply(rel_trans)
        
        # Store results
        abs_poses[i+1, :3] = curr_pos
        abs_poses[i+1, 3:7] = curr_rot.as_quat()
    
    return abs_poses

def trajectory_integration_unwrapped_trans(rel_poses):
    """
    Integrate relative poses but using global translations (unrotated).
    
    Args:
        rel_poses: Array of relative poses [N, 6]
        
    Returns:
        abs_poses: Array of absolute poses [N+1, 7]
    """
    # First pose is identity
    abs_poses = np.zeros((len(rel_poses) + 1, 7))
    abs_poses[0, 6] = 1.0  # Identity quaternion [0, 0, 0, 1]
    
    # Current state
    curr_pos = np.zeros(3)
    curr_rot = Rotation.identity()
    
    for i in range(len(rel_poses)):
        # Get relative pose
        rel_euler = rel_poses[i, :3]  # roll, pitch, yaw
        rel_trans = rel_poses[i, 3:6]  # x, y, z
        
        # Convert to rotation object using XYZ convention
        rel_rot = Rotation.from_euler('xyz', rel_euler)
        
        # Update current rotation: curr_rot = curr_rot * rel_rot
        curr_rot = curr_rot * rel_rot
        
        # Add translation directly without rotation
        curr_pos = curr_pos + rel_trans
        
        # Store results
        abs_poses[i+1, :3] = curr_pos
        abs_poses[i+1, 3:7] = curr_rot.as_quat()
    
    return abs_poses

def visualize_trajectories(trajectories, labels, title="Trajectory Comparison"):
    """
    Visualize multiple trajectories for comparison.
    
    Args:
        trajectories: List of trajectory arrays [N, 3] or [N, 7]
        labels: List of labels for each trajectory
        title: Plot title
    """
    # Create figure with 3D and 2D views
    fig = plt.figure(figsize=(15, 10))
    
    # 3D view
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.set_title("3D View")
    
    # 2D views (top, side, front)
    ax2 = fig.add_subplot(222)
    ax2.set_title("Top View (XY)")
    
    ax3 = fig.add_subplot(223)
    ax3.set_title("Side View (XZ)")
    
    ax4 = fig.add_subplot(224)
    ax4.set_title("Front View (YZ)")
    
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
    
    for i, (traj, label) in enumerate(zip(trajectories, labels)):
        # Extract positions (might be [N, 3] or [N, 7])
        if traj.shape[1] > 3:
            # [N, 7] with [x, y, z, qx, qy, qz, qw]
            positions = traj[:, :3]
        else:
            # [N, 3] with [x, y, z]
            positions = traj
        
        color = colors[i % len(colors)]
        
        # 3D plot
        ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], color=color, label=label)
        ax1.scatter(positions[0, 0], positions[0, 1], positions[0, 2], color=color, marker='o', s=50)
        
        # 2D plots
        ax2.plot(positions[:, 0], positions[:, 1], color=color, label=label)
        ax2.scatter(positions[0, 0], positions[0, 1], color=color, marker='o', s=50)
        
        ax3.plot(positions[:, 0], positions[:, 2], color=color, label=label)
        ax3.scatter(positions[0, 0], positions[0, 2], color=color, marker='o', s=50)
        
        ax4.plot(positions[:, 1], positions[:, 2], color=color, label=label)
        ax4.scatter(positions[0, 1], positions[0, 2], color=color, marker='o', s=50)
    
    # Set labels
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.grid(True)
    
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Z (m)')
    ax3.grid(True)
    
    ax4.set_xlabel('Y (m)')
    ax4.set_ylabel('Z (m)')
    ax4.grid(True)
    
    # Add legend
    ax1.legend()
    ax2.legend()
    
    # Set overall title
    fig.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    return fig

def compute_ground_truth_relative_poses(gt_poses):
    """
    Compute relative poses from ground truth absolute poses.
    
    Args:
        gt_poses: Array of ground truth poses [N, 6] with [roll, pitch, yaw, x, y, z]
        
    Returns:
        rel_poses: Array of relative poses [N-1, 6]
    """
    n_poses = len(gt_poses)
    rel_poses = np.zeros((n_poses - 1, 6))
    
    for i in range(n_poses - 1):
        # Get consecutive poses
        pose1 = gt_poses[i]
        pose2 = gt_poses[i + 1]
        
        # Extract Euler angles and positions
        euler1 = pose1[:3]
        euler2 = pose2[:3]
        pos1 = pose1[3:6]
        pos2 = pose2[3:6]
        
        # Convert to rotation matrices
        rot1 = Rotation.from_euler('xyz', euler1)
        rot2 = Rotation.from_euler('xyz', euler2)
        
        # Compute relative rotation: R_rel = R1^-1 * R2
        rot_rel = rot1.inv() * rot2
        euler_rel = rot_rel.as_euler('xyz')
        
        # Compute relative translation: t_rel = R1^-1 * (t2 - t1)
        trans_rel = rot1.inv().apply(pos2 - pos1)
        
        # Store relative pose
        rel_poses[i, :3] = euler_rel
        rel_poses[i, 3:6] = trans_rel
    
    return rel_poses

def scale_translations(rel_poses, scale_factor):
    """
    Scale the translation part of relative poses.
    
    Args:
        rel_poses: Array of relative poses [N, 6]
        scale_factor: Scale factor to apply
    
    Returns:
        scaled_poses: Array of scaled relative poses [N, 6]
    """
    scaled_poses = rel_poses.copy()
    scaled_poses[:, 3:6] *= scale_factor
    return scaled_poses

def main():
    parser = argparse.ArgumentParser(description="Debug trajectory integration")
    parser.add_argument("--pred_rel_poses", type=str, required=True, help="File with predicted relative poses")
    parser.add_argument("--gt_poses", type=str, required=True, help="File with ground truth poses")
    parser.add_argument("--output", type=str, default="debug_output", help="Output directory")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale factor for translations")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load predicted relative poses
    pred_rel_poses = np.loadtxt(args.pred_rel_poses)
    logger.info(f"Loaded {len(pred_rel_poses)} predicted relative poses")
    
    # Log statistics of the predictions
    trans_mean = np.mean(np.linalg.norm(pred_rel_poses[:, 3:6], axis=1))
    trans_std = np.std(np.linalg.norm(pred_rel_poses[:, 3:6], axis=1))
    rot_mean = np.mean(np.linalg.norm(pred_rel_poses[:, :3], axis=1))
    rot_std = np.std(np.linalg.norm(pred_rel_poses[:, :3], axis=1))
    
    logger.info(f"Translation magnitude: {trans_mean:.4f} ± {trans_std:.4f} m")
    logger.info(f"Rotation magnitude: {rot_mean:.4f} ± {rot_std:.4f} rad")
    
    # Load ground truth poses (use np.load for npy files)
    gt_poses = np.load(args.gt_poses)
    logger.info(f"Loaded {len(gt_poses)} ground truth poses")
    
    # Compute ground truth relative poses for comparison
    gt_rel_poses = compute_ground_truth_relative_poses(gt_poses)
    logger.info(f"Computed {len(gt_rel_poses)} ground truth relative poses")
    
    # Log statistics of ground truth relative poses
    gt_trans_mean = np.mean(np.linalg.norm(gt_rel_poses[:, 3:6], axis=1))
    gt_trans_std = np.std(np.linalg.norm(gt_rel_poses[:, 3:6], axis=1))
    gt_rot_mean = np.mean(np.linalg.norm(gt_rel_poses[:, :3], axis=1))
    gt_rot_std = np.std(np.linalg.norm(gt_rel_poses[:, :3], axis=1))
    
    logger.info(f"GT Translation magnitude: {gt_trans_mean:.4f} ± {gt_trans_std:.4f} m")
    logger.info(f"GT Rotation magnitude: {gt_rot_mean:.4f} ± {gt_rot_std:.4f} rad")
    
    # Compare the first few relative poses
    logger.info("First 5 predicted relative poses:")
    for i in range(min(5, len(pred_rel_poses))):
        logger.info(f"  {i}: {pred_rel_poses[i]}")
    
    logger.info("First 5 ground truth relative poses:")
    for i in range(min(5, len(gt_rel_poses))):
        logger.info(f"  {i}: {gt_rel_poses[i]}")
    
    # Check for scale difference
    scale_factor = gt_trans_mean / trans_mean if trans_mean > 0 else 1.0
    logger.info(f"Translation scale factor (GT/Pred): {scale_factor:.4f}")
    
    # If a scale was specified, apply it
    if args.scale != 1.0:
        logger.info(f"Applying user-specified scale factor of {args.scale}")
        scaled_pred_rel_poses = scale_translations(pred_rel_poses, args.scale)
    else:
        # Apply computed scale factor if it's reasonable
        if 0.1 <= scale_factor <= 10.0:
            logger.info(f"Applying computed scale factor of {scale_factor:.4f}")
            scaled_pred_rel_poses = scale_translations(pred_rel_poses, scale_factor)
        else:
            logger.warning(f"Computed scale factor {scale_factor:.4f} is extreme, not applying")
            scaled_pred_rel_poses = pred_rel_poses
    
    # Also try an inverse scale - maybe translations are backward?
    inverse_scaled_poses = scale_translations(pred_rel_poses, -scale_factor)
    logger.info(f"Created inverse-scaled poses for testing")
    
    # Create a version with translations divided by 100
    div100_poses = scale_translations(pred_rel_poses, 0.01)
    logger.info(f"Created poses with translations divided by 100")
    
    # Create a version with translations divided by 1000
    div1000_poses = scale_translations(pred_rel_poses, 0.001)
    logger.info(f"Created poses with translations divided by 1000")
    
    # Integrate trajectories with different methods
    logger.info("Integrating trajectories with different methods...")
    
    # 1. Simple accumulation (just add translations)
    simple_positions = simple_trajectory_integration(pred_rel_poses)
    logger.info(f"Simple integration complete: {simple_positions.shape}")
    
    # 2. ZYX Euler convention
    zyx_traj = trajectory_integration_zyx(pred_rel_poses)
    logger.info(f"ZYX integration complete: {zyx_traj.shape}")
    
    # 3. XYZ Euler convention
    xyz_traj = trajectory_integration_xyz(pred_rel_poses)
    logger.info(f"XYZ integration complete: {xyz_traj.shape}")
    
    # 4. Scaled ZYX
    scaled_zyx_traj = trajectory_integration_zyx(scaled_pred_rel_poses)
    logger.info(f"Scaled ZYX integration complete: {scaled_zyx_traj.shape}")
    
    # 5. Inverse scaled XYZ
    inv_xyz_traj = trajectory_integration_xyz(inverse_scaled_poses)
    logger.info(f"Inverse scaled XYZ integration complete: {inv_xyz_traj.shape}")
    
    # 6. Division by 100
    div100_traj = trajectory_integration_xyz(div100_poses)
    logger.info(f"Division by 100 integration complete: {div100_traj.shape}")
    
    # 7. Division by 1000
    div1000_traj = trajectory_integration_xyz(div1000_poses)
    logger.info(f"Division by 1000 integration complete: {div1000_traj.shape}")
    
    # 8. Try integration with unrotated translations
    unwrapped_traj = trajectory_integration_unwrapped_trans(pred_rel_poses)
    logger.info(f"Unrotated translations integration complete: {unwrapped_traj.shape}")
    
    # 9. Convert ground truth poses to standard format for visualization
    gt_traj = np.zeros((len(gt_poses), 7))
    gt_traj[:, :3] = gt_poses[:, 3:6]  # Positions
    
    # Convert Euler to quaternions
    for i in range(len(gt_poses)):
        euler = gt_poses[i, :3]
        quat = Rotation.from_euler('xyz', euler).as_quat()
        gt_traj[i, 3:7] = quat
    
    # Visualize all trajectories in batches to avoid overcrowding
    # First batch: main methods
    trajectories1 = [gt_traj, simple_positions, xyz_traj, zyx_traj, scaled_zyx_traj]
    labels1 = ['Ground Truth', 'Simple Sum', 'XYZ Integration', 'ZYX Integration', f'Scaled ZYX (x{args.scale if args.scale != 1.0 else scale_factor:.4f})']
    
    # Create visualization
    fig1 = visualize_trajectories(trajectories1, labels1, title="Trajectory Integration Comparison - Main Methods")
    plt.savefig(os.path.join(args.output, "trajectory_comparison_main.png"), dpi=300)
    plt.close(fig1)
    logger.info(f"Main methods visualization saved")
    
    # Second batch: scale experiments
    trajectories2 = [gt_traj, xyz_traj, inv_xyz_traj, div100_traj, div1000_traj, unwrapped_traj]
    labels2 = ['Ground Truth', 'XYZ Integration', f'Inverse Scaled (x-{scale_factor:.4f})', 'Div by 100', 'Div by 1000', 'Unrotated Trans']
    
    fig2 = visualize_trajectories(trajectories2, labels2, title="Trajectory Integration Comparison - Scale Experiments")
    plt.savefig(os.path.join(args.output, "trajectory_comparison_scales.png"), dpi=300)
    plt.close(fig2)
    logger.info(f"Scale experiments visualization saved")
    
    # Also try a reduced trajectory (first N points)
    n_points = min(100, len(pred_rel_poses))
    
    # Integrate reduced trajectories
    reduced_simple = simple_trajectory_integration(pred_rel_poses[:n_points])
    reduced_xyz = trajectory_integration_xyz(pred_rel_poses[:n_points])
    reduced_zyx = trajectory_integration_zyx(pred_rel_poses[:n_points])
    reduced_div100 = trajectory_integration_xyz(div100_poses[:n_points])
    reduced_div1000 = trajectory_integration_xyz(div1000_poses[:n_points])
    
    # Visualize reduced trajectories
    reduced_trajectories = [
        gt_traj[:n_points+1], 
        reduced_simple, 
        reduced_xyz, 
        reduced_zyx, 
        reduced_div100,
        reduced_div1000
    ]
    
    reduced_labels = [
        'Ground Truth', 
        'Simple Sum', 
        'XYZ Integration', 
        'ZYX Integration', 
        'Div by 100',
        'Div by 1000'
    ]
    
    fig3 = visualize_trajectories(
        reduced_trajectories, 
        reduced_labels, 
        title=f"Reduced Trajectory Comparison (First {n_points} Points)"
    )
    plt.savefig(os.path.join(args.output, "reduced_trajectory_comparison.png"), dpi=300)
    plt.close(fig3)
    logger.info(f"Reduced trajectory visualization saved")
    
    # Save the most promising trajectory method for further analysis
    most_promising = "div1000_traj"  # You can change this based on results
    np.savetxt(os.path.join(args.output, "most_promising_trajectory.txt"), eval(most_promising), fmt="%.6f")
    logger.info(f"Saved most promising trajectory method: {most_promising}")
    
    # Calculate and save errors for each method
    error_summary = {}
    
    # Function to calculate mean error
    def calc_error(pred, gt):
        min_len = min(len(pred), len(gt))
        return np.mean(np.linalg.norm(pred[:min_len, :3] - gt[:min_len, :3], axis=1))
    
    error_summary["simple"] = calc_error(simple_positions, gt_traj)
    error_summary["xyz"] = calc_error(xyz_traj, gt_traj)
    error_summary["zyx"] = calc_error(zyx_traj, gt_traj)
    error_summary["scaled_zyx"] = calc_error(scaled_zyx_traj, gt_traj)
    error_summary["inv_xyz"] = calc_error(inv_xyz_traj, gt_traj)
    error_summary["div100"] = calc_error(div100_traj, gt_traj)
    error_summary["div1000"] = calc_error(div1000_traj, gt_traj)
    error_summary["unwrapped"] = calc_error(unwrapped_traj, gt_traj)
    
    # Find best method
    best_method = min(error_summary.items(), key=lambda x: x[1])
    logger.info(f"Best method: {best_method[0]} with error {best_method[1]:.4f} m")
    
    # Log all errors
    logger.info("Errors for all methods:")
    for method, error in error_summary.items():
        logger.info(f"  {method}: {error:.4f} m")
    
    # Save error summary
    with open(os.path.join(args.output, "error_summary.txt"), "w") as f:
        f.write("Method,Error(m)\n")
        for method, error in error_summary.items():
            f.write(f"{method},{error:.6f}\n")
    
    logger.info(f"All results saved to {args.output} directory")
    return 0

if __name__ == "__main__":
    exit(main())