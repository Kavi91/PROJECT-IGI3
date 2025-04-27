#!/usr/bin/env python3
"""
3D Trajectory Visualization

This script creates an interactive 3D plot of trajectory data, allowing you
to visualize and compare predicted, ground truth, and aligned trajectories.

python plot_3d_trajectory.py --pred_traj results/Kite_training_sunny_trajectory_0001_20250507-011404/pred_rel_poses.txt --gt_traj /media/krkavinda/New\ Volume/Mid-Air-Dataset/MidAir_processed/Kite_training/sunny/poses/camera_poses_0001.npy --aligned_traj segment_aligned_results/aligned_trajectory.txt --output 3d_comparison.png

"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_trajectory(file_path):
    """Load trajectory data from file."""
    try:
        # Try to load as text file first
        try:
            data = np.loadtxt(file_path)
            logger.info(f"Loaded trajectory from text file: {file_path}, shape: {data.shape}")
            return data
        except:
            # Try to load as numpy file
            data = np.load(file_path)
            logger.info(f"Loaded trajectory from numpy file: {file_path}, shape: {data.shape}")
            return data
    except Exception as e:
        logger.error(f"Error loading trajectory from {file_path}: {e}")
        return None

def convert_gt_format(gt_poses):
    """Convert ground truth poses to standard format if needed."""
    if gt_poses.shape[1] == 6:  # [roll, pitch, yaw, x, y, z] format
        # Convert to [x, y, z, ...] format
        traj = np.zeros((len(gt_poses), 7))
        traj[:, :3] = gt_poses[:, 3:6]  # Positions
        logger.info("Converted ground truth from [roll, pitch, yaw, x, y, z] to [x, y, z, qx, qy, qz, qw]")
        return traj
    return gt_poses

def plot_3d_trajectory(pred_traj, gt_traj, aligned_traj=None, output_path="3d_trajectory.png", title="3D Trajectory Comparison", show_plot=True):
    """
    Create a 3D plot of trajectories.
    
    Args:
        pred_traj: Predicted trajectory [N, 7] or [N, 3]
        gt_traj: Ground truth trajectory [N, 7] or [N, 3]
        aligned_traj: Aligned trajectory [N, 7] or [N, 3] (optional)
        output_path: Path to save the figure
        title: Plot title
        show_plot: Whether to show the plot interactively
    """
    # Extract positions if needed
    if pred_traj.shape[1] > 3:
        pred_pos = pred_traj[:, :3]
    else:
        pred_pos = pred_traj
    
    if gt_traj.shape[1] > 3:
        gt_pos = gt_traj[:, :3]
    else:
        gt_pos = gt_traj
    
    if aligned_traj is not None:
        if aligned_traj.shape[1] > 3:
            aligned_pos = aligned_traj[:, :3]
        else:
            aligned_pos = aligned_traj
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectories
    ax.plot(pred_pos[:, 0], pred_pos[:, 1], pred_pos[:, 2], 'r-', linewidth=2, label='Predicted')
    ax.plot(gt_pos[:, 0], gt_pos[:, 1], gt_pos[:, 2], 'b-', linewidth=2, label='Ground Truth')
    
    if aligned_traj is not None:
        ax.plot(aligned_pos[:, 0], aligned_pos[:, 1], aligned_pos[:, 2], 'g-', linewidth=2, label='Aligned')
    
    # Add start/end markers
    ax.scatter(pred_pos[0, 0], pred_pos[0, 1], pred_pos[0, 2], c='g', marker='o', s=100, label='Start')
    ax.scatter(pred_pos[-1, 0], pred_pos[-1, 1], pred_pos[-1, 2], c='r', marker='o', s=100, label='End (Pred)')
    ax.scatter(gt_pos[-1, 0], gt_pos[-1, 1], gt_pos[-1, 2], c='b', marker='o', s=100, label='End (GT)')
    
    if aligned_traj is not None:
        ax.scatter(aligned_pos[-1, 0], aligned_pos[-1, 1], aligned_pos[-1, 2], c='g', marker='s', s=100, label='End (Aligned)')
    
    # Add arrows to show direction along trajectory
    sample_freq = max(1, len(gt_pos) // 20)  # Show ~20 arrows
    for i in range(0, len(gt_pos)-1, sample_freq):
        # Ground truth direction
        dx = gt_pos[i+1, 0] - gt_pos[i, 0]
        dy = gt_pos[i+1, 1] - gt_pos[i, 1]
        dz = gt_pos[i+1, 2] - gt_pos[i, 2]
        
        # Normalize and scale for visibility
        mag = np.sqrt(dx**2 + dy**2 + dz**2)
        if mag > 0:
            scale = min(1.0, mag) / mag
            ax.quiver(gt_pos[i, 0], gt_pos[i, 1], gt_pos[i, 2], 
                     dx * scale, dy * scale, dz * scale, 
                     color='blue', alpha=0.6, arrow_length_ratio=0.15)
    
    # Set labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    
    # Set equal aspect ratio
    max_range_gt = max([
        np.max(gt_pos[:, 0]) - np.min(gt_pos[:, 0]),
        np.max(gt_pos[:, 1]) - np.min(gt_pos[:, 1]),
        np.max(gt_pos[:, 2]) - np.min(gt_pos[:, 2])
    ])
    
    # Set view limits based on ground truth
    mid_x_gt = (np.max(gt_pos[:, 0]) + np.min(gt_pos[:, 0])) / 2
    mid_y_gt = (np.max(gt_pos[:, 1]) + np.min(gt_pos[:, 1])) / 2
    mid_z_gt = (np.max(gt_pos[:, 2]) + np.min(gt_pos[:, 2])) / 2
    
    # Add padding
    ax.set_xlim(mid_x_gt - max_range_gt/1.5, mid_x_gt + max_range_gt/1.5)
    ax.set_ylim(mid_y_gt - max_range_gt/1.5, mid_y_gt + max_range_gt/1.5)
    ax.set_zlim(mid_z_gt - max_range_gt/1.5, mid_z_gt + max_range_gt/1.5)
    
    # Add legend
    ax.legend()
    
    # Add grid
    ax.grid(True)
    
    # Set initial viewpoint
    ax.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved 3D trajectory plot to: {output_path}")
    
    if show_plot:
        plt.show()
    
    return fig

def create_animation(pred_traj, gt_traj, aligned_traj=None, output_path="trajectory_animation.mp4", fps=30, duration=10):
    """
    Create an animation rotating around the 3D trajectory.
    
    Args:
        pred_traj: Predicted trajectory [N, 7] or [N, 3]
        gt_traj: Ground truth trajectory [N, 7] or [N, 3]
        aligned_traj: Aligned trajectory [N, 7] or [N, 3] (optional)
        output_path: Path to save the animation
        fps: Frames per second
        duration: Animation duration in seconds
    """
    # Extract positions if needed
    if pred_traj.shape[1] > 3:
        pred_pos = pred_traj[:, :3]
    else:
        pred_pos = pred_traj
    
    if gt_traj.shape[1] > 3:
        gt_pos = gt_traj[:, :3]
    else:
        gt_pos = gt_traj
    
    if aligned_traj is not None:
        if aligned_traj.shape[1] > 3:
            aligned_pos = aligned_traj[:, :3]
        else:
            aligned_pos = aligned_traj
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectories
    ax.plot(pred_pos[:, 0], pred_pos[:, 1], pred_pos[:, 2], 'r-', linewidth=2, label='Predicted')
    ax.plot(gt_pos[:, 0], gt_pos[:, 1], gt_pos[:, 2], 'b-', linewidth=2, label='Ground Truth')
    
    if aligned_traj is not None:
        ax.plot(aligned_pos[:, 0], aligned_pos[:, 1], aligned_pos[:, 2], 'g-', linewidth=2, label='Aligned')
    
    # Add start/end markers
    ax.scatter(pred_pos[0, 0], pred_pos[0, 1], pred_pos[0, 2], c='g', marker='o', s=100, label='Start')
    ax.scatter(pred_pos[-1, 0], pred_pos[-1, 1], pred_pos[-1, 2], c='r', marker='o', s=100, label='End (Pred)')
    ax.scatter(gt_pos[-1, 0], gt_pos[-1, 1], gt_pos[-1, 2], c='b', marker='o', s=100, label='End (GT)')
    
    if aligned_traj is not None:
        ax.scatter(aligned_pos[-1, 0], aligned_pos[-1, 1], aligned_pos[-1, 2], c='g', marker='s', s=100, label='End (Aligned)')
    
    # Set labels
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title("3D Trajectory Comparison")
    
    # Set equal aspect ratio
    max_range_gt = max([
        np.max(gt_pos[:, 0]) - np.min(gt_pos[:, 0]),
        np.max(gt_pos[:, 1]) - np.min(gt_pos[:, 1]),
        np.max(gt_pos[:, 2]) - np.min(gt_pos[:, 2])
    ])
    
    # Set view limits based on ground truth
    mid_x_gt = (np.max(gt_pos[:, 0]) + np.min(gt_pos[:, 0])) / 2
    mid_y_gt = (np.max(gt_pos[:, 1]) + np.min(gt_pos[:, 1])) / 2
    mid_z_gt = (np.max(gt_pos[:, 2]) + np.min(gt_pos[:, 2])) / 2
    
    # Add padding
    ax.set_xlim(mid_x_gt - max_range_gt/1.5, mid_x_gt + max_range_gt/1.5)
    ax.set_ylim(mid_y_gt - max_range_gt/1.5, mid_y_gt + max_range_gt/1.5)
    ax.set_zlim(mid_z_gt - max_range_gt/1.5, mid_z_gt + max_range_gt/1.5)
    
    # Add legend
    ax.legend()
    
    # Add grid
    ax.grid(True)
    
    # Animation function
    frames = int(fps * duration)
    azimuth_angles = np.linspace(0, 360, frames)
    
    def update(frame):
        ax.view_init(elev=30, azim=azimuth_angles[frame])
        return fig,
    
    # Create animation
    ani = FuncAnimation(fig, update, frames=frames, blit=True)
    
    # Save animation
    try:
        ani.save(output_path, fps=fps, dpi=150, writer='ffmpeg')
        logger.info(f"Saved animation to: {output_path}")
    except Exception as e:
        logger.error(f"Error saving animation: {e}")
        logger.info("Trying to save with pillow writer instead...")
        try:
            ani.save(output_path.replace('.mp4', '.gif'), fps=fps//2, dpi=100, writer='pillow')
            logger.info(f"Saved animation as GIF to: {output_path.replace('.mp4', '.gif')}")
        except Exception as e2:
            logger.error(f"Error saving animation as GIF: {e2}")
    
    plt.close(fig)
    return ani

def crop_to_segment(pred_traj, gt_traj, aligned_traj=None, start_idx=0, end_idx=None):
    """
    Crop trajectories to a specific segment.
    
    Args:
        pred_traj: Predicted trajectory [N, D]
        gt_traj: Ground truth trajectory [N, D]
        aligned_traj: Aligned trajectory [N, D] (optional)
        start_idx: Start index
        end_idx: End index
        
    Returns:
        Cropped trajectories
    """
    # Set default end index
    if end_idx is None:
        end_idx = min(len(pred_traj), len(gt_traj))
    
    # Crop trajectories
    pred_traj_crop = pred_traj[start_idx:end_idx]
    gt_traj_crop = gt_traj[start_idx:end_idx]
    
    if aligned_traj is not None:
        aligned_traj_crop = aligned_traj[start_idx:end_idx]
        return pred_traj_crop, gt_traj_crop, aligned_traj_crop
    
    return pred_traj_crop, gt_traj_crop

def main():
    parser = argparse.ArgumentParser(description="Plot 3D trajectory visualization")
    parser.add_argument("--pred_traj", type=str, required=True, help="Predicted trajectory file")
    parser.add_argument("--gt_traj", type=str, required=True, help="Ground truth trajectory file")
    parser.add_argument("--aligned_traj", type=str, default=None, help="Aligned trajectory file (optional)")
    parser.add_argument("--output", type=str, default="3d_trajectory.png", help="Output image path")
    parser.add_argument("--title", type=str, default="3D Trajectory Comparison", help="Plot title")
    parser.add_argument("--animate", action="store_true", help="Create animation")
    parser.add_argument("--no_display", action="store_true", help="Don't display the plot")
    parser.add_argument("--segment", type=str, default=None, help="Segment to plot (start:end)")
    args = parser.parse_args()
    
    # Load trajectory data
    pred_traj = load_trajectory(args.pred_traj)
    gt_traj = load_trajectory(args.gt_traj)
    
    if pred_traj is None or gt_traj is None:
        logger.error("Failed to load trajectory data")
        return 1
    
    # Load aligned trajectory if provided
    aligned_traj = None
    if args.aligned_traj:
        aligned_traj = load_trajectory(args.aligned_traj)
        if aligned_traj is None:
            logger.warning(f"Failed to load aligned trajectory from {args.aligned_traj}")
    
    # Convert ground truth format if needed
    gt_traj = convert_gt_format(gt_traj)
    
    # Ensure trajectories have the same length
    min_len = min(len(pred_traj), len(gt_traj))
    pred_traj = pred_traj[:min_len]
    gt_traj = gt_traj[:min_len]
    if aligned_traj is not None:
        aligned_traj = aligned_traj[:min_len]
    
    # Crop to segment if specified
    if args.segment:
        try:
            start_idx, end_idx = map(int, args.segment.split(':'))
            if aligned_traj is not None:
                pred_traj, gt_traj, aligned_traj = crop_to_segment(pred_traj, gt_traj, aligned_traj, start_idx, end_idx)
            else:
                pred_traj, gt_traj = crop_to_segment(pred_traj, gt_traj, None, start_idx, end_idx)
            logger.info(f"Cropped trajectories to segment [{start_idx}:{end_idx}]")
        except Exception as e:
            logger.error(f"Error parsing segment '{args.segment}': {e}")
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create 3D plot
    plot_3d_trajectory(
        pred_traj, 
        gt_traj, 
        aligned_traj=aligned_traj,
        output_path=args.output,
        title=args.title,
        show_plot=not args.no_display
    )
    
    # Create animation if requested
    if args.animate:
        animation_path = os.path.join(
            os.path.dirname(args.output),
            os.path.splitext(os.path.basename(args.output))[0] + "_animation.mp4"
        )
        create_animation(
            pred_traj, 
            gt_traj, 
            aligned_traj=aligned_traj,
            output_path=animation_path
        )
    
    return 0

if __name__ == "__main__":
    sys.exit(main())