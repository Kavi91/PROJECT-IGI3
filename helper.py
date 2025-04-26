import numpy as np
import torch
import math

def normalize_angle(angle):
    """Normalize angle to [-pi, pi]"""
    return (angle + np.pi) % (2 * np.pi) - np.pi

def euler_to_rotation_matrix(euler_angles):
    """
    Convert euler angles (roll, pitch, yaw) to rotation matrix
    Using ZYX convention (yaw, pitch, roll)
    """
    roll, pitch, yaw = euler_angles
    
    # Precompute trig values
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    
    # Construct rotation matrix
    R = np.zeros((3, 3))
    R[0, 0] = cy * cp
    R[0, 1] = cy * sp * sr - sy * cr
    R[0, 2] = cy * sp * cr + sy * sr
    R[1, 0] = sy * cp
    R[1, 1] = sy * sp * sr + cy * cr
    R[1, 2] = sy * sp * cr - cy * sr
    R[2, 0] = -sp
    R[2, 1] = cp * sr
    R[2, 2] = cp * cr
    
    return R

def rotation_matrix_to_euler(R):
    """
    Convert rotation matrix to euler angles (roll, pitch, yaw)
    Using ZYX convention (yaw, pitch, roll)
    """
    pitch = -np.arcsin(R[2, 0])
    
    if abs(R[2, 0]) > 0.99999:
        # Gimbal lock case
        yaw = 0.0
        roll = np.arctan2(R[0, 1], R[1, 1])
    else:
        # Normal case
        yaw = np.arctan2(R[1, 0], R[0, 0])
        roll = np.arctan2(R[2, 1], R[2, 2])
    
    return np.array([roll, pitch, yaw])

def relative_to_absolute_pose(relative_poses, body_to_camera):
    """
    Integrate relative poses to absolute trajectory
    
    Args:
        relative_poses: Tensor of shape [seq_len, 6] with [roll, pitch, yaw, x, y, z]
        body_to_camera: 3x3 rotation matrix for camera-to-body transformation
    
    Returns:
        absolute_poses: Tensor of shape [seq_len+1, 6]
    """
    device = relative_poses.device
    seq_len = relative_poses.shape[0]
    
    # Initialize absolute trajectory with zero pose
    absolute_poses = [torch.zeros(6, dtype=torch.float32, device=device)]
    
    # Current state
    current_R = torch.eye(3, dtype=torch.float32, device=device)
    current_t = torch.zeros(3, dtype=torch.float32, device=device)
    
    # Camera to body transformation
    camera_to_body = torch.inverse(body_to_camera)
    
    for i in range(seq_len):
        rel_pose = relative_poses[i]
        rel_angles = rel_pose[:3].cpu().numpy()
        rel_t_camera = rel_pose[3:]
        
        # Convert to rotation matrix and transform to body frame
        R_rel_camera = torch.tensor(euler_to_rotation_matrix(rel_angles), dtype=torch.float32, device=device)
        R_rel_body = camera_to_body @ R_rel_camera @ body_to_camera
        
        # Update rotation and position
        current_R = current_R @ R_rel_body
        rel_t_body = camera_to_body @ rel_t_camera
        delta_t_world = current_R @ rel_t_body
        current_t = current_t + delta_t_world
        
        # Convert to euler angles
        current_angles = torch.tensor(rotation_matrix_to_euler(current_R.cpu().numpy()), 
                                     dtype=torch.float32, device=device)
        
        # Normalize angles
        current_angles = torch.tensor([normalize_angle(angle) for angle in current_angles], 
                                      dtype=torch.float32, device=device)
        
        # Create and append absolute pose
        absolute_pose = torch.cat((current_angles, current_t))
        absolute_poses.append(absolute_pose)
    
    return torch.stack(absolute_poses)

def compute_ate(pred_poses, gt_poses):
    """
    Compute Absolute Trajectory Error (ATE)
    
    Args:
        pred_poses: Predicted trajectory [N, 6]
        gt_poses: Ground truth trajectory [N, 6]
        
    Returns:
        ate_mean: Mean ATE
        ate_std: STD of ATE
    """
    # Extract positions
    pred_pos = pred_poses[:, 3:6]
    gt_pos = gt_poses[:, 3:6]
    
    # Compute position errors
    errors = torch.norm(pred_pos - gt_pos, dim=1)
    
    return errors.mean().item(), errors.std().item()

def compute_rpe(pred_poses, gt_poses):
    """
    Compute Relative Pose Error (RPE)
    
    Args:
        pred_poses: Predicted trajectory [N, 6]
        gt_poses: Ground truth trajectory [N, 6]
        
    Returns:
        rpe_trans_mean: Mean translational RPE
        rpe_trans_std: STD of translational RPE
        rpe_rot_mean: Mean rotational RPE
        rpe_rot_std: STD of rotational RPE
    """
    trans_errors = []
    rot_errors = []
    
    for i in range(pred_poses.shape[0] - 1):
        # Translation error
        pred_delta_t = pred_poses[i+1, 3:6] - pred_poses[i, 3:6]
        gt_delta_t = gt_poses[i+1, 3:6] - gt_poses[i, 3:6]
        trans_error = torch.norm(pred_delta_t - gt_delta_t)
        trans_errors.append(trans_error.item())
        
        # Rotation error
        pred_R1 = torch.tensor(euler_to_rotation_matrix(pred_poses[i, :3].cpu().numpy()))
        pred_R2 = torch.tensor(euler_to_rotation_matrix(pred_poses[i+1, :3].cpu().numpy()))
        gt_R1 = torch.tensor(euler_to_rotation_matrix(gt_poses[i, :3].cpu().numpy()))
        gt_R2 = torch.tensor(euler_to_rotation_matrix(gt_poses[i+1, :3].cpu().numpy()))
        
        pred_delta_R = pred_R1.t() @ pred_R2
        gt_delta_R = gt_R1.t() @ gt_R2
        error_R = gt_delta_R.t() @ pred_delta_R
        
        # Compute angle from rotation matrix (error_R)
        trace = error_R.trace()
        if trace > 3 - 1e-10:
            angle_error = 0.0
        else:
            angle_error = np.arccos(torch.clamp((trace - 1) / 2, -1.0, 1.0).item())
        
        rot_errors.append(angle_error)
    
    trans_errors = np.array(trans_errors)
    rot_errors = np.array(rot_errors)
    
    return (np.mean(trans_errors), np.std(trans_errors),
            np.mean(rot_errors), np.std(rot_errors))

def visualize_trajectory(pred_traj, gt_traj=None, title="Trajectory Visualization", save_path=None):
    """
    Visualize the predicted (and optionally ground truth) trajectories
    
    Args:
        pred_traj: Predicted trajectory [N, 6]
        gt_traj: Ground truth trajectory [N, 6]
        title: Plot title
        save_path: Path to save the figure
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # Convert to numpy if tensors
    if isinstance(pred_traj, torch.Tensor):
        pred_traj = pred_traj.cpu().numpy()
    if gt_traj is not None and isinstance(gt_traj, torch.Tensor):
        gt_traj = gt_traj.cpu().numpy()
    
    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot predicted trajectory
    ax.plot(pred_traj[:, 3], pred_traj[:, 4], pred_traj[:, 5], 'r-', linewidth=2, label='Predicted')
    ax.scatter(pred_traj[0, 3], pred_traj[0, 4], pred_traj[0, 5], c='r', marker='o', s=100)
    
    # If ground truth is provided, plot it too
    if gt_traj is not None:
        min_len = min(len(pred_traj), len(gt_traj))
        ax.plot(gt_traj[:min_len, 3], gt_traj[:min_len, 4], gt_traj[:min_len, 5], 'g-', linewidth=2, label='Ground Truth')
        ax.scatter(gt_traj[0, 3], gt_traj[0, 4], gt_traj[0, 5], c='g', marker='o', s=100)
    
    # Set labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    ax.legend()
    
    # Adjust view to show the best perspective
    if gt_traj is not None:
        all_x = np.concatenate((pred_traj[:, 3], gt_traj[:, 3]))
        all_y = np.concatenate((pred_traj[:, 4], gt_traj[:, 4]))
        all_z = np.concatenate((pred_traj[:, 5], gt_traj[:, 5]))
    else:
        all_x = pred_traj[:, 3]
        all_y = pred_traj[:, 4]
        all_z = pred_traj[:, 5]
    
    max_range = max(all_x.max() - all_x.min(), 
                   all_y.max() - all_y.min(),
                   all_z.max() - all_z.min()) / 2.0
    
    mid_x = (all_x.max() + all_x.min()) / 2
    mid_y = (all_y.max() + all_y.min()) / 2
    mid_z = (all_z.max() + all_z.min()) / 2
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Save plot if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig, ax