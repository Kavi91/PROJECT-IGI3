#!/usr/bin/env python3
"""
Trajectory Alignment Script using Segment-Based Umeyama Method

This script aligns a predicted trajectory to ground truth using segment-based Umeyama alignment,
which applies separate transformations to different segments of the trajectory.

python segment_alignment.py --pred_rel_poses /home/krkavinda/PROJECT-IGI3/results/Kite_training_sunny_trajectory_0008_20250507-090939/pred_rel_poses.txt --gt_poses /media/krkavinda/New\ Volume/Mid-Air-Dataset/MidAir_processed/Kite_training/sunny/poses/camera_poses_0008.npy --scale 0.5 --no_rotation --segment_size 50 --overlap 25 --output segment_aligned_results


"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import logging
import json
from tqdm import tqdm

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
    scale = trace_SD / var_X if var_X > 1e-10 else 1.0
    
    # Compute translation
    t = mu_Y - scale * R @ mu_X
    
    return R, t, scale

def align_trajectory_segment(pred_traj, gt_traj, start_idx, end_idx):
    """
    Align a segment of the predicted trajectory to ground truth.
    
    Args:
        pred_traj: Predicted trajectory [N, 7]
        gt_traj: Ground truth trajectory [N, 7]
        start_idx: Start index of segment
        end_idx: End index of segment
        
    Returns:
        R: Rotation matrix [3, 3]
        t: Translation vector [3]
        s: Scale factor
    """
    # Extract segment
    pred_seg = pred_traj[start_idx:end_idx, :3]
    gt_seg = gt_traj[start_idx:end_idx, :3]
    
    # Check if segment is large enough
    if len(pred_seg) < 3:
        # Return identity transformation if segment is too small
        return np.eye(3), np.zeros(3), 1.0
    
    # Compute alignment transformation
    try:
        R, t, s = umeyama_alignment(pred_seg, gt_seg)
        return R, t, s
    except Exception as e:
        logger.warning(f"Error aligning segment [{start_idx}:{end_idx}]: {e}")
        # Return identity transformation on error
        return np.eye(3), np.zeros(3), 1.0

def align_trajectory_segments(pred_traj, gt_traj, segment_size=30, overlap=10):
    """
    Align trajectory segments and combine them smoothly.
    
    Args:
        pred_traj: Predicted trajectory [N, 7]
        gt_traj: Ground truth trajectory [N, 7]
        segment_size: Size of each segment in frames
        overlap: Overlap between segments in frames
        
    Returns:
        aligned_traj: Aligned trajectory [N, 7]
    """
    # Ensure trajectory lengths match
    min_len = min(len(pred_traj), len(gt_traj))
    pred_traj = pred_traj[:min_len]
    gt_traj = gt_traj[:min_len]
    
    # Initialize aligned trajectory
    aligned_traj = np.zeros_like(pred_traj)
    
    # Initialize weight matrix for smooth blending
    weights = np.zeros(min_len)
    
    # Get segment start indices
    stride = segment_size - overlap
    segment_starts = list(range(0, min_len - segment_size + 1, stride))
    
    # If the last segment doesn't reach the end, add one more segment
    if segment_starts[-1] + segment_size < min_len:
        segment_starts.append(min_len - segment_size)
    
    # Process each segment
    logger.info(f"Processing {len(segment_starts)} segments...")
    for i, start_idx in enumerate(tqdm(segment_starts)):
        # Compute end index
        end_idx = min(start_idx + segment_size, min_len)
        
        # Align segment
        R, t, s = align_trajectory_segment(pred_traj, gt_traj, start_idx, end_idx)
        
        # Apply transformation to segment positions
        seg_aligned_pos = s * (R @ pred_traj[start_idx:end_idx, :3].T).T + t
        
        # Apply rotation to orientations
        R_rot = Rotation.from_matrix(R)
        seg_aligned_rot = np.zeros((end_idx - start_idx, 4))
        
        for j in range(start_idx, end_idx):
            idx = j - start_idx
            # Check if quaternion is valid before creating rotation object
            if np.linalg.norm(pred_traj[j, 3:7]) > 1e-10:
                # Normalize quaternion to ensure it's valid
                quat = pred_traj[j, 3:7] / np.linalg.norm(pred_traj[j, 3:7])
                pred_rot = Rotation.from_quat(quat)
                seg_aligned_rot[idx] = (R_rot * pred_rot).as_quat()
            else:
                # Use identity quaternion if original is invalid
                seg_aligned_rot[idx] = np.array([0, 0, 0, 1])
        
        # Create weight function for smooth blending
        segment_weights = np.zeros(end_idx - start_idx)
        for j in range(end_idx - start_idx):
            # Normalized position in segment [0, 1]
            pos = j / (end_idx - start_idx - 1) if end_idx > start_idx + 1 else 0.5
            # Weight is higher in middle, lower at edges
            segment_weights[j] = 0.5 - 0.5 * np.cos(2 * np.pi * pos)
        
        # Apply transformation with weights
        for j in range(start_idx, end_idx):
            idx = j - start_idx
            pos_weight = segment_weights[idx]
            
            # Add weighted transformation to position
            aligned_traj[j, :3] += pos_weight * seg_aligned_pos[idx]
            aligned_traj[j, 3:7] += pos_weight * seg_aligned_rot[idx]
            
            # Add to total weight
            weights[j] += pos_weight
    
    # Normalize by weights
    for i in range(min_len):
        if weights[i] > 0:
            aligned_traj[i, :3] /= weights[i]
            
            # Normalize quaternion
            quat_norm = np.linalg.norm(aligned_traj[i, 3:7])
            if quat_norm > 1e-10:
                aligned_traj[i, 3:7] /= quat_norm
            else:
                # Use identity quaternion if invalid
                aligned_traj[i, 3:7] = np.array([0, 0, 0, 1])
    
    return aligned_traj

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
        
        # Extract rotations - check for valid quaternions
        try:
            # Normalize quaternions to be safe
            p1_quat = p1[3:7] / np.linalg.norm(p1[3:7]) if np.linalg.norm(p1[3:7]) > 1e-10 else np.array([0, 0, 0, 1])
            p2_quat = p2[3:7] / np.linalg.norm(p2[3:7]) if np.linalg.norm(p2[3:7]) > 1e-10 else np.array([0, 0, 0, 1])
            g1_quat = g1[3:7] / np.linalg.norm(g1[3:7]) if np.linalg.norm(g1[3:7]) > 1e-10 else np.array([0, 0, 0, 1])
            g2_quat = g2[3:7] / np.linalg.norm(g2[3:7]) if np.linalg.norm(g2[3:7]) > 1e-10 else np.array([0, 0, 0, 1])
            
            p1_rot = Rotation.from_quat(p1_quat)
            p2_rot = Rotation.from_quat(p2_quat)
            g1_rot = Rotation.from_quat(g1_quat)
            g2_rot = Rotation.from_quat(g2_quat)
            
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
        except Exception as e:
            # Skip this frame if there's an error with quaternions
            continue
    
    if not trans_errors:
        return 0, 0  # Return zeros if no valid errors were computed
        
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
    parser = argparse.ArgumentParser(description="Align trajectory using segment-based Umeyama method")
    parser.add_argument("--pred_rel_poses", type=str, required=True, help="Predicted relative poses file")
    parser.add_argument("--gt_poses", type=str, required=True, help="Ground truth poses file")
    parser.add_argument("--scale", type=float, default=0.5, help="Initial scale factor for integration")
    parser.add_argument("--no_rotation", action="store_true", help="Don't apply rotation in integration")
    parser.add_argument("--segment_size", type=int, default=50, help="Size of segments in frames")
    parser.add_argument("--overlap", type=int, default=25, help="Overlap between segments in frames")
    parser.add_argument("--output", type=str, default="segment_aligned_results", help="Output directory")
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
    
    # Align trajectory with segment-based approach
    logger.info(f"Aligning trajectory using segment-based approach with segment_size={args.segment_size}, overlap={args.overlap}...")
    aligned_traj = align_trajectory_segments(
        pred_traj, 
        gt_traj, 
        segment_size=args.segment_size,
        overlap=args.overlap
    )
    
    # Compute ATE after alignment
    aligned_ate_mean, aligned_ate_std, aligned_ate_max = compute_ate(aligned_traj, gt_traj)
    logger.info(f"ATE after segment-based alignment: {aligned_ate_mean:.4f} ± {aligned_ate_std:.4f} m (max: {aligned_ate_max:.4f} m)")
    
    # Compute RPE after alignment
    aligned_rpe_trans, aligned_rpe_rot = compute_rpe(aligned_traj, gt_traj)
    logger.info(f"RPE after segment-based alignment: Trans={aligned_rpe_trans:.4f} m, Rot={aligned_rpe_rot:.4f} rad ({aligned_rpe_rot * 180 / np.pi:.4f} deg)")
    
    # Plot trajectory after alignment
    fig_after = plot_trajectory(
        pred_traj, 
        gt_traj, 
        aligned_traj=aligned_traj,
        title=f"After Segment-Based Alignment: ATE={aligned_ate_mean:.4f}±{aligned_ate_std:.4f} m",
        save_path=os.path.join(args.output, "trajectory_after_alignment.png")
    )
    plt.close(fig_after)
    
    # Save results
    np.savetxt(os.path.join(args.output, "aligned_trajectory.txt"), aligned_traj, fmt="%.8f")
    
    # Save alignment metrics
    metrics = {
        "segment_alignment": {
            "segment_size": args.segment_size,
            "overlap": args.overlap,
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
    }
    
    with open(os.path.join(args.output, "alignment_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    
    logger.info("Segment-based alignment complete. Results saved to %s", args.output)
    
    # Try different segment sizes to find the optimal one if requested
    if args.segment_size == 0:  # Special flag to try different sizes
        logger.info("Trying different segment sizes to find optimal parameters...")
        
        segment_sizes = [10, 20, 30, 50, 100, 200, 500]
        best_ate = float('inf')
        best_params = None
        results = []
        
        for seg_size in segment_sizes:
            # Set overlap to half the segment size
            overlap = seg_size // 2
            
            logger.info(f"Testing segment_size={seg_size}, overlap={overlap}...")
            
            # Align trajectory
            seg_aligned_traj = align_trajectory_segments(
                pred_traj, 
                gt_traj, 
                segment_size=seg_size,
                overlap=overlap
            )
            
            # Compute ATE
            seg_ate_mean, seg_ate_std, seg_ate_max = compute_ate(seg_aligned_traj, gt_traj)
            
            # Store results
            results.append({
                "segment_size": seg_size,
                "overlap": overlap,
                "ate_mean": seg_ate_mean,
                "ate_std": seg_ate_std,
                "ate_max": seg_ate_max
            })
            
            logger.info(f"  ATE: {seg_ate_mean:.4f} ± {seg_ate_std:.4f} m")
            
            # Check if this is better
            if seg_ate_mean < best_ate:
                best_ate = seg_ate_mean
                best_params = {
                    "segment_size": seg_size,
                    "overlap": overlap
                }
                
                # Plot and save best result
                fig_best = plot_trajectory(
                    pred_traj, 
                    gt_traj, 
                    aligned_traj=seg_aligned_traj,
                    title=f"Best Segment Alignment (size={seg_size}): ATE={seg_ate_mean:.4f}±{seg_ate_std:.4f} m",
                    save_path=os.path.join(args.output, f"trajectory_seg_size_{seg_size}.png")
                )
                plt.close(fig_best)
                
                # Save best trajectory
                np.savetxt(os.path.join(args.output, f"aligned_trajectory_seg_size_{seg_size}.txt"), seg_aligned_traj, fmt="%.8f")
        
        # Save parameter search results
        results.sort(key=lambda x: x["ate_mean"])
        
        logger.info("Segment size parameter search results:")
        for i, result in enumerate(results):
            logger.info(f"  {i+1}. Segment Size={result['segment_size']}, Overlap={result['overlap']}, ATE={result['ate_mean']:.4f}±{result['ate_std']:.4f} m")
        
        logger.info(f"Best segment-based alignment parameters: segment_size={best_params['segment_size']}, overlap={best_params['overlap']}, ATE={best_ate:.4f} m")
        
        with open(os.path.join(args.output, "segment_size_search.json"), "w") as f:
            json.dump(results, f, indent=4)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())