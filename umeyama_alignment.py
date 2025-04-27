#!/usr/bin/env python3
"""
Trajectory Alignment Script using Umeyama Method

This script aligns a predicted trajectory to ground truth using the Umeyama method,
which finds the optimal similarity transformation (rotation, translation, and scale).

python umeyama_alignment.py --pred_rel_poses results/Kite_training_sunny_trajectory_0001_20250507-011404/pred_rel_poses.txt --gt_poses /media/krkavinda/New\ Volume/Mid-Air-Dataset/MidAir_processed/Kite_training/sunny/poses/camera_poses_0001.npy --scale 0.5 --no_rotation --output aligned_results



"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_trajectory(file_path):
    """
    Load trajectory from file.
    
    Args:
        file_path: Path to trajectory file
        
    Returns:
        Trajectory array [N, 7] with [x, y, z, qx, qy, qz, qw]
    """
    try:
        data = np.loadtxt(file_path)
        if data.shape[1] == 7:  # Already in [x, y, z, qx, qy, qz, qw] format
            return data
        elif data.shape[1] == 6:  # [roll, pitch, yaw, x, y, z] format
            # Convert to standard format
            traj = np.zeros((len(data), 7))
            traj[:, :3] = data[:, 3:6]  # Positions
            
            # Convert Euler to quaternions
            for i in range(len(data)):
                euler = data[i, :3]
                quat = Rotation.from_euler('xyz', euler).as_quat()
                traj[i, 3:7] = quat
            
            return traj
        else:
            logger.error(f"Unsupported trajectory format with {data.shape[1]} columns")
            return None
    except Exception as e:
        logger.error(f"Error loading trajectory from {file_path}: {e}")
        return None

def simple_trajectory_integration(rel_poses, scale_factor=0.5, apply_rotation=False):
    """
    Integrate relative poses to get absolute trajectory.
    
    Args:
        rel_poses: Array of relative poses [N, 6] with [roll, pitch, yaw, x, y, z]
        scale_factor: Scaling factor for translations
        apply_rotation: Whether to apply rotation to translations
        
    Returns:
        abs_poses: Array of absolute poses [N+1, 7] with [x, y, z, qx, qy, qz, qw]
    """
    # Create absolute poses array (first pose is identity)
    abs_poses = np.zeros((len(rel_poses) + 1, 7))
    abs_poses[0, 6] = 1.0  # Identity quaternion [0, 0, 0, 1]
    
    # Current state
    curr_pos = np.zeros(3)
    curr_rot = Rotation.identity()
    
    for i in range(len(rel_poses)):
        # Get relative pose
        rel_euler = rel_poses[i, :3]  # roll, pitch, yaw
        rel_trans = rel_poses[i, 3:6]  # x, y, z
        
        # Scale translations
        rel_trans = rel_trans * scale_factor
        
        # Update rotation
        rel_rot = Rotation.from_euler('xyz', rel_euler)
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

def umeyama_alignment(X, Y):
    """
    Umeyama algorithm to find the optimal similarity transformation
    (rotation, translation, and scale) that aligns point sets X to Y.
    
    Args:
        X: Source points [N, D]
        Y: Target points [N, D]
        
    Returns:
        R: Rotation matrix [D, D]
        t: Translation vector [D]
        s: Scale factor
        
    Reference:
        Umeyama, "Least-Squares Estimation of Transformation Parameters
        Between Two Point Patterns", IEEE PAMI, 1991
    """
    # Ensure X and Y have the same dimensions
    assert X.shape == Y.shape, "X and Y must have the same shape"
    
    # Get dimensions
    n, m = X.shape
    
    # Center the points
    mu_X = np.mean(X, axis=0)
    mu_Y = np.mean(Y, axis=0)
    X0 = X - mu_X
    Y0 = Y - mu_Y
    
    # Compute variance of X
    var_X = np.sum(np.square(X0)) / n
    
    # Compute covariance matrix
    Sigma = X0.T @ Y0 / n
    
    # Singular Value Decomposition
    U, D, Vt = np.linalg.svd(Sigma)
    
    # Determine if we need to correct the rotation matrix to ensure a
    # right-handed coordinate system
    S = np.eye(m)
    if np.linalg.det(Sigma) < 0 or (m == 3 and np.linalg.det(U @ Vt) < 0):
        S[m-1, m-1] = -1
    
    # Compute rotation matrix
    R = U @ S @ Vt
    
    # Compute scale factor
    trace_SD = np.sum(S * D)
    scale = trace_SD / var_X
    
    # Compute translation
    t = mu_Y - scale * R @ mu_X
    
    return R, t, scale

def align_trajectory(pred_traj, gt_traj):
    """
    Align predicted trajectory to ground truth using Umeyama method.
    
    Args:
        pred_traj: Predicted trajectory [N, 7] with [x, y, z, qx, qy, qz, qw]
        gt_traj: Ground truth trajectory [N, 7] with [x, y, z, qx, qy, qz, qw]
        
    Returns:
        aligned_traj: Aligned trajectory [N, 7]
        R: Rotation matrix [3, 3]
        t: Translation vector [3]
        s: Scale factor
    """
    # Ensure same length
    min_len = min(len(pred_traj), len(gt_traj))
    pred_traj = pred_traj[:min_len]
    gt_traj = gt_traj[:min_len]
    
    # Extract positions
    pred_pos = pred_traj[:, :3]
    gt_pos = gt_traj[:, :3]
    
    # Compute alignment transformation
    R, t, s = umeyama_alignment(pred_pos, gt_pos)
    
    # Apply transformation to predicted positions
    aligned_pos = s * (R @ pred_pos.T).T + t
    
    # Apply rotation to predicted orientations
    aligned_rot = np.zeros((min_len, 4))
    R_rot = Rotation.from_matrix(R)
    
    for i in range(min_len):
        pred_rot = Rotation.from_quat(pred_traj[i, 3:7])
        aligned_rot[i] = (R_rot * pred_rot).as_quat()
    
    # Combine into aligned trajectory
    aligned_traj = np.zeros((min_len, 7))
    aligned_traj[:, :3] = aligned_pos
    aligned_traj[:, 3:7] = aligned_rot
    
    return aligned_traj, R, t, s

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

def plot_trajectory(pred_traj, gt_traj, aligned_traj=None, title="Trajectory Comparison", save_path=None):
    """
    Plot trajectory comparison.
    
    Args:
        pred_traj: Predicted trajectory [N, 7]
        gt_traj: Ground truth trajectory [N, 7]
        aligned_traj: Aligned trajectory [N, 7] (optional)
        title: Plot title
        save_path: Path to save the figure
    """
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # Extract positions
    pred_pos = pred_traj[:, :3]
    gt_pos = gt_traj[:, :3]
    
    # XY plane (top view)
    axs[0].plot(pred_pos[:, 0], pred_pos[:, 1], 'r-', linewidth=2, label='Predicted')
    axs[0].plot(gt_pos[:, 0], gt_pos[:, 1], 'b-', linewidth=2, label='Ground Truth')
    
    # XZ plane (side view)
    axs[1].plot(pred_pos[:, 0], pred_pos[:, 2], 'r-', linewidth=2, label='Predicted')
    axs[1].plot(gt_pos[:, 0], gt_pos[:, 2], 'b-', linewidth=2, label='Ground Truth')
    
    # YZ plane (front view)
    axs[2].plot(pred_pos[:, 1], pred_pos[:, 2], 'r-', linewidth=2, label='Predicted')
    axs[2].plot(gt_pos[:, 1], gt_pos[:, 2], 'b-', linewidth=2, label='Ground Truth')
    
    # Plot aligned trajectory if provided
    if aligned_traj is not None:
        aligned_pos = aligned_traj[:, :3]
        axs[0].plot(aligned_pos[:, 0], aligned_pos[:, 1], 'g-', linewidth=2, label='Aligned')
        axs[1].plot(aligned_pos[:, 0], aligned_pos[:, 2], 'g-', linewidth=2, label='Aligned')
        axs[2].plot(aligned_pos[:, 1], aligned_pos[:, 2], 'g-', linewidth=2, label='Aligned')
    
    # Add start/end markers
    for ax in axs:
        if pred_pos.shape[0] > 0:
            ax.scatter(pred_pos[0, 0] if ax == axs[0] or ax == axs[1] else pred_pos[0, 1], 
                      pred_pos[0, 2] if ax == axs[1] or ax == axs[2] else pred_pos[0, 1], 
                      c='g', marker='o', s=100, label='Start')
            ax.scatter(pred_pos[-1, 0] if ax == axs[0] or ax == axs[1] else pred_pos[-1, 1], 
                      pred_pos[-1, 2] if ax == axs[1] or ax == axs[2] else pred_pos[-1, 1], 
                      c='r', marker='o', s=100, label='End (Pred)')
        
        if gt_pos.shape[0] > 0:
            ax.scatter(gt_pos[-1, 0] if ax == axs[0] or ax == axs[1] else gt_pos[-1, 1], 
                      gt_pos[-1, 2] if ax == axs[1] or ax == axs[2] else gt_pos[-1, 1], 
                      c='b', marker='o', s=100, label='End (GT)')
        
        if aligned_traj is not None and aligned_pos.shape[0] > 0:
            ax.scatter(aligned_pos[-1, 0] if ax == axs[0] or ax == axs[1] else aligned_pos[-1, 1], 
                      aligned_pos[-1, 2] if ax == axs[1] or ax == axs[2] else aligned_pos[-1, 1], 
                      c='g', marker='o', s=100, label='End (Aligned)')
    
    # Set labels
    axs[0].set_xlabel('X (m)')
    axs[0].set_ylabel('Y (m)')
    axs[0].set_title('XY Plane (Top View)')
    axs[0].grid(True)
    axs[0].legend()
    
    axs[1].set_xlabel('X (m)')
    axs[1].set_ylabel('Z (m)')
    axs[1].set_title('XZ Plane (Side View)')
    axs[1].grid(True)
    
    axs[2].set_xlabel('Y (m)')
    axs[2].set_ylabel('Z (m)')
    axs[2].set_title('YZ Plane (Front View)')
    axs[2].grid(True)
    
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    
    return fig

def main():
    parser = argparse.ArgumentParser(description="Align trajectory using Umeyama method")
    parser.add_argument("--pred_rel_poses", type=str, required=True, help="Predicted relative poses file")
    parser.add_argument("--gt_poses", type=str, required=True, help="Ground truth poses file")
    parser.add_argument("--scale", type=float, default=0.5, help="Initial scale factor for integration")
    parser.add_argument("--no_rotation", action="store_true", help="Don't apply rotation in integration")
    parser.add_argument("--output", type=str, default="aligned_results", help="Output directory")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load predicted relative poses
    pred_rel_poses = np.loadtxt(args.pred_rel_poses)
    logger.info(f"Loaded {len(pred_rel_poses)} predicted relative poses")
    
    # Load ground truth poses
    gt_poses = np.load(args.gt_poses)
    logger.info(f"Loaded {len(gt_poses)} ground truth poses")
    
    # Convert ground truth to standard format
    gt_traj = np.zeros((len(gt_poses), 7))
    gt_traj[:, :3] = gt_poses[:, 3:6]  # Positions
    
    # Convert Euler to quaternions
    for i in range(len(gt_poses)):
        euler = gt_poses[i, :3]
        quat = Rotation.from_euler('xyz', euler).as_quat()
        gt_traj[i, 3:7] = quat
    
    # Integrate predicted trajectory
    logger.info(f"Integrating trajectory with scale={args.scale}, apply_rotation={not args.no_rotation}")
    pred_traj = simple_trajectory_integration(
        pred_rel_poses, 
        scale_factor=args.scale, 
        apply_rotation=not args.no_rotation
    )
    
    # Compute ATE before alignment
    ate_mean, ate_std, ate_max = compute_ate(pred_traj, gt_traj)
    logger.info(f"ATE before alignment: {ate_mean:.4f} ± {ate_std:.4f} m (max: {ate_max:.4f} m)")
    
    # Compute RPE before alignment
    rpe_trans, rpe_rot = compute_rpe(pred_traj, gt_traj)
    logger.info(f"RPE before alignment: Trans={rpe_trans:.4f} m, Rot={rpe_rot:.4f} rad ({rpe_rot * 180 / np.pi:.4f} deg)")
    
    # Plot trajectory before alignment
    fig_before = plot_trajectory(
        pred_traj, 
        gt_traj, 
        title=f"Before Alignment: ATE={ate_mean:.4f}±{ate_std:.4f} m",
        save_path=os.path.join(args.output, "trajectory_before_alignment.png")
    )
    plt.close(fig_before)
    
    # Align trajectory
    logger.info("Aligning trajectory using Umeyama method...")
    aligned_traj, R, t, s = align_trajectory(pred_traj, gt_traj)
    logger.info(f"Alignment transformation: scale={s:.4f}")
    logger.info(f"Rotation matrix:\n{R}")
    logger.info(f"Translation vector: {t}")
    
    # Compute ATE after alignment
    aligned_ate_mean, aligned_ate_std, aligned_ate_max = compute_ate(aligned_traj, gt_traj)
    logger.info(f"ATE after alignment: {aligned_ate_mean:.4f} ± {aligned_ate_std:.4f} m (max: {aligned_ate_max:.4f} m)")
    
    # Compute RPE after alignment
    aligned_rpe_trans, aligned_rpe_rot = compute_rpe(aligned_traj, gt_traj)
    logger.info(f"RPE after alignment: Trans={aligned_rpe_trans:.4f} m, Rot={aligned_rpe_rot:.4f} rad ({aligned_rpe_rot * 180 / np.pi:.4f} deg)")
    
    # Plot trajectory after alignment
    fig_after = plot_trajectory(
        pred_traj, 
        gt_traj, 
        aligned_traj=aligned_traj,
        title=f"After Alignment: ATE={aligned_ate_mean:.4f}±{aligned_ate_std:.4f} m",
        save_path=os.path.join(args.output, "trajectory_after_alignment.png")
    )
    plt.close(fig_after)
    
    # Save results
    np.savetxt(os.path.join(args.output, "aligned_trajectory.txt"), aligned_traj, fmt="%.8f")
    
    # Save alignment transformation
    transformation = {
        "scale": float(s),
        "rotation_matrix": R.tolist(),
        "translation_vector": t.tolist(),
        "metrics": {
            "before_alignment": {
                "ate_mean": float(ate_mean),
                "ate_std": float(ate_std),
                "ate_max": float(ate_max),
                "rpe_trans": float(rpe_trans),
                "rpe_rot": float(rpe_rot),
                "rpe_rot_deg": float(rpe_rot * 180 / np.pi)
            },
            "after_alignment": {
                "ate_mean": float(aligned_ate_mean),
                "ate_std": float(aligned_ate_std),
                "ate_max": float(aligned_ate_max),
                "rpe_trans": float(aligned_rpe_trans),
                "rpe_rot": float(aligned_rpe_rot),
                "rpe_rot_deg": float(aligned_rpe_rot * 180 / np.pi)
            }
        }
    }
    
    with open(os.path.join(args.output, "alignment_transformation.json"), "w") as f:
        json.dump(transformation, f, indent=4)
    
    logger.info("Alignment complete. Results saved to %s", args.output)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())