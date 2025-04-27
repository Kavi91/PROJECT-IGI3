import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
from scipy.spatial.transform import Rotation

###########################################
# Utility Functions for Transformations
###########################################

def normalize_angle_delta(angle):
    """
    Normalize an angle to the range [-pi, pi].
    """
    angle = angle % (2 * np.pi)
    angle = np.where(angle > np.pi, angle - 2 * np.pi, angle)
    angle = np.where(angle < -np.pi, angle + 2 * np.pi, angle)
    return angle

def euler_angles_to_rotation_matrix(euler_angles):
    """
    Convert ZYX Euler angles to rotation matrix (NED frame).
    
    Args:
        euler_angles (np.ndarray): Array with [roll, pitch, yaw] in radians (ZYX order).
    
    Returns:
        np.ndarray: 3x3 rotation matrix
    """
    # Handle both single angles and batch of angles
    single_input = len(euler_angles.shape) == 1
    if single_input:
        euler_angles = euler_angles[np.newaxis, :]
    
    roll, pitch, yaw = euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2]
    
    # Compute trigonometric values
    cos_r, sin_r = np.cos(roll), np.sin(roll)
    cos_p, sin_p = np.cos(pitch), np.sin(pitch)
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)

    # Initialize rotation matrices
    R = np.zeros((euler_angles.shape[0], 3, 3), dtype=np.float32)
    
    # ZYX rotation matrix (yaw -> pitch -> roll) in NED frame
    R[:, 0, 0] = cos_y * cos_p
    R[:, 0, 1] = cos_y * sin_p * sin_r - sin_y * cos_r
    R[:, 0, 2] = cos_y * sin_p * cos_r + sin_y * sin_r
    R[:, 1, 0] = sin_y * cos_p
    R[:, 1, 1] = sin_y * sin_p * sin_r + cos_y * cos_r
    R[:, 1, 2] = sin_y * sin_p * cos_r - cos_y * sin_r
    R[:, 2, 0] = -sin_p
    R[:, 2, 1] = cos_p * sin_r
    R[:, 2, 2] = cos_p * cos_r

    # Ensure orthogonality
    for i in range(euler_angles.shape[0]):
        U, _, Vt = np.linalg.svd(R[i])
        R[i] = U @ Vt
    
    return R[0] if single_input else R

def rotation_matrix_to_euler_angles(R):
    """
    Extract ZYX Euler angles from rotation matrix (NED frame).
    
    Args:
        R (np.ndarray): 3x3 rotation matrix
    
    Returns:
        np.ndarray: [roll, pitch, yaw] Euler angles in ZYX order (radians)
    """
    # Check if input is a single matrix or batch of matrices
    single_input = len(R.shape) == 2
    if single_input:
        R = R[np.newaxis, :, :]
    
    euler = np.zeros((R.shape[0], 3), dtype=np.float32)
    
    for i in range(R.shape[0]):
        # Ensure orthogonality
        U, _, Vt = np.linalg.svd(R[i])
        R_ortho = U @ Vt
        
        # Extract pitch (y-axis rotation)
        pitch = -np.arcsin(R_ortho[2, 0])
        
        # Check for gimbal lock
        if np.abs(R_ortho[2, 0]) > 0.99999:
            # Gimbal lock case
            euler[i, 2] = np.arctan2(-R_ortho[1, 2], R_ortho[1, 1])  # yaw
            euler[i, 0] = 0  # roll is arbitrary in gimbal lock, set to 0
        else:
            # Normal case
            euler[i, 0] = np.arctan2(R_ortho[2, 1], R_ortho[2, 2])  # roll
            euler[i, 2] = np.arctan2(R_ortho[1, 0], R_ortho[0, 0])  # yaw
        
        euler[i, 1] = pitch  # pitch
    
    # Normalize angles
    euler = normalize_angle_delta(euler)
    
    return euler[0] if single_input else euler

def compute_relative_pose(pose1, pose2):
    """
    Compute relative pose from pose1 to pose2 in NED coordinate system.
    
    Args:
        pose1 (np.ndarray): [roll1, pitch1, yaw1, x1, y1, z1]
        pose2 (np.ndarray): [roll2, pitch2, yaw2, x2, y2, z2]
    
    Returns:
        np.ndarray: Relative pose [d_roll, d_pitch, d_yaw, d_x, d_y, d_z]
    """
    # Extract rotations and translations
    R1 = euler_angles_to_rotation_matrix(pose1[:3])
    R2 = euler_angles_to_rotation_matrix(pose2[:3])
    t1 = pose1[3:].reshape(3, 1)
    t2 = pose2[3:].reshape(3, 1)
    
    # Compute relative rotation (R_rel = R1^T * R2)
    R_rel = R1.T @ R2
    
    # Convert to Euler angles
    euler_rel = rotation_matrix_to_euler_angles(R_rel)
    
    # Compute relative translation in world frame
    t_rel_world = t2 - t1
    
    # Transform to body1 frame
    t_rel_body = R1.T @ t_rel_world
    
    # Combine into a single array
    rel_pose = np.concatenate((euler_rel, t_rel_body.flatten()))
    
    return rel_pose

def transform_to_camera_frame(body_pose, body_to_camera, body_to_camera_translation):
    """
    Transform a pose from body frame to camera frame, including translation offset.
    
    Args:
        body_pose (np.ndarray): [roll, pitch, yaw, x, y, z] in body frame
        body_to_camera (np.ndarray): 3x3 rotation matrix from body to camera frame
        body_to_camera_translation (np.ndarray): 3D translation vector from body to camera (in meters)
    
    Returns:
        np.ndarray: [roll, pitch, yaw, x, y, z] in camera frame
    """
    # Extract rotation and translation
    R_body = euler_angles_to_rotation_matrix(body_pose[:3])
    t_body = body_pose[3:].reshape(3, 1)
    
    # Transform rotation to camera frame
    R_camera = body_to_camera @ R_body
    
    # Transform translation to camera frame, including the offset
    t_camera = body_to_camera @ t_body + body_to_camera_translation.reshape(3, 1)
    
    # Convert rotation to Euler angles
    euler_camera = rotation_matrix_to_euler_angles(R_camera)
    
    # Combine into a single array
    camera_pose = np.concatenate((euler_camera, t_camera.flatten()))
    
    return camera_pose

def camera_to_body_frame(camera_pose, body_to_camera, body_to_camera_translation):
    """
    Transform a pose from camera frame back to body frame, including translation offset.
    
    Args:
        camera_pose (np.ndarray): [roll, pitch, yaw, x, y, z] in camera frame
        body_to_camera (np.ndarray): 3x3 rotation matrix from body to camera frame
        body_to_camera_translation (np.ndarray): 3D translation vector from body to camera (in meters)
    
    Returns:
        np.ndarray: [roll, pitch, yaw, x, y, z] in body frame
    """
    # Compute camera to body transformation
    camera_to_body = body_to_camera.T
    
    # Extract rotation and translation
    R_camera = euler_angles_to_rotation_matrix(camera_pose[:3])
    t_camera = camera_pose[3:].reshape(3, 1)
    
    # Transform rotation to body frame
    R_body = camera_to_body @ R_camera
    
    # Transform translation to body frame, accounting for the offset
    t_body = camera_to_body @ (t_camera - body_to_camera_translation.reshape(3, 1))
    
    # Convert rotation to Euler angles
    euler_body = rotation_matrix_to_euler_angles(R_body)
    
    # Combine into a single array
    body_pose = np.concatenate((euler_body, t_body.flatten()))
    
    return body_pose

def plot_coordinate_frame(ax, pose, scale=1.0):
    """
    Plot a coordinate frame at the given pose.
    
    Args:
        ax: Matplotlib 3D axis
        pose (np.ndarray): [roll, pitch, yaw, x, y, z]
        scale (float): Scale factor for the coordinate frame visualization
    """
    origin = pose[3:6]
    R = euler_angles_to_rotation_matrix(pose[:3])
    
    # X, Y, Z axes
    for i, color in enumerate(['r', 'g', 'b']):
        ax.quiver(origin[0], origin[1], origin[2],
                  R[0, i] * scale, R[1, i] * scale, R[2, i] * scale,
                  color=color, linewidth=1.5)

def validate_transformation_reversibility(poses, body_to_camera, body_to_camera_translation, output_file=None):
    """
    Validate that transformations between body and camera frames are reversible.
    
    Args:
        poses (np.ndarray): Array of poses [N, 6] with [roll, pitch, yaw, x, y, z]
        body_to_camera (np.ndarray): 3x3 rotation matrix from body to camera frame
        body_to_camera_translation (np.ndarray): 3D translation vector from body to camera (in meters)
        output_file (str, optional): Path to save the output. If None, print to console.
    """
    output = []
    output.append("----- Transformation Reversibility Test -----\n")
    
    max_errors = np.zeros(6)
    mean_errors = np.zeros(6)
    
    for i in range(len(poses)):
        # Original pose in body frame
        original_pose = poses[i]
        
        # Transform to camera frame
        camera_pose = transform_to_camera_frame(original_pose, body_to_camera, body_to_camera_translation)
        
        # Transform back to body frame
        recovered_pose = camera_to_body_frame(camera_pose, body_to_camera, body_to_camera_translation)
        
        # Compute errors
        errors = np.abs(original_pose - recovered_pose)
        
        # Normalize angle errors
        for j in range(3):
            if errors[j] > np.pi:
                errors[j] = 2 * np.pi - errors[j]
        
        max_errors = np.maximum(max_errors, errors)
        mean_errors += errors
    
    mean_errors /= len(poses)
    
    output.append("Maximum transformation roundtrip errors:")
    for i, label in enumerate(['Roll', 'Pitch', 'Yaw', 'X', 'Y', 'Z']):
        if i < 3:
            output.append(f"{label}: {max_errors[i]:.6f} rad ({np.degrees(max_errors[i]):.4f} deg)")
        else:
            output.append(f"{label}: {max_errors[i]:.6f} m")
    
    output.append("\nMean transformation roundtrip errors:")
    for i, label in enumerate(['Roll', 'Pitch', 'Yaw', 'X', 'Y', 'Z']):
        if i < 3:
            output.append(f"{label}: {mean_errors[i]:.6f} rad ({np.degrees(mean_errors[i]):.4f} deg)")
        else:
            output.append(f"{label}: {max_errors[i]:.6f} m")
    
    if np.any(max_errors[:3] > 0.01) or np.any(max_errors[3:] > 0.01):
        output.append("\nWarning: Significant transformation errors detected. Check your transformation matrix.")
    else:
        output.append("\nTransformation reversibility test passed. Errors are within acceptable limits.")
    
    # Write to file or print
    if output_file:
        with open(output_file, 'w') as f:
            f.write('\n'.join(output))
    else:
        print('\n'.join(output))
        
def plot_ground_truth_absolute_vs_camera_relative(poses, body_to_camera, body_to_camera_translation, output_dir='.'):
    """
    Create a professional side-by-side 3D plot of ground truth absolute trajectory and
    camera frame relative translations.
    
    Args:
        poses (np.ndarray): Array of poses [N, 6] with [roll, pitch, yaw, x, y, z]
        body_to_camera (np.ndarray): 3x3 rotation matrix from body to camera frame
        body_to_camera_translation (np.ndarray): 3D translation vector from body to camera (in meters)
        output_dir (str): Directory to save output visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Compute relative poses
    relative_poses = []
    for i in range(1, len(poses)):
        rel_pose = compute_relative_pose(poses[i-1], poses[i])
        relative_poses.append(rel_pose)
    
    relative_poses = np.array(relative_poses)
    
    # Transform relative poses to camera frame
    camera_relative_poses = []
    for i in range(len(relative_poses)):
        camera_rel = transform_to_camera_frame(relative_poses[i], body_to_camera, body_to_camera_translation)
        camera_relative_poses.append(camera_rel)
    
    camera_relative_poses = np.array(camera_relative_poses)
    
    # Create a figure with specific size
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 8,
        'axes.labelsize': 8,
        'axes.titlesize': 9,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'lines.linewidth': 0.5,
        'legend.fontsize': 7,
        'legend.frameon': False,
        'figure.dpi': 300
    })
    
    # Solid blue color for both plots
    blue_color = '#1f77b4'
    
    fig = plt.figure(figsize=(15/2.54, 8/2.54))
    
    # Create side-by-side 3D plots
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Plot absolute ground truth trajectory
    # Adjust ground truth positions to the camera frame
    gt_positions = poses[:, 3:6]
    gt_positions_camera = gt_positions + body_to_camera_translation
    ax1.plot(gt_positions_camera[:, 0], gt_positions_camera[:, 1], gt_positions_camera[:, 2], 
             c=blue_color, linewidth=0.5, label='Trajectory')
    ax1.scatter(gt_positions_camera[0, 0], gt_positions_camera[0, 1], gt_positions_camera[0, 2], 
                c='green', marker='o', s=20, label='Start')
    ax1.scatter(gt_positions_camera[-1, 0], gt_positions_camera[-1, 1], gt_positions_camera[-1, 2], 
                c='red', marker='o', s=20, label='End')
    
    # Configure first plot
    ax1.set_xlabel('X (m)', labelpad=1)
    ax1.set_ylabel('Y (m)', labelpad=1)
    ax1.set_zlabel('Z (m)', labelpad=1)
    ax1.set_title('Ground Truth\nAbsolute Trajectory (NED)')
    ax1.grid(False)
    ax1.legend(loc='upper right', fontsize=6)
    
    # Extract just the XYZ positions from camera relative poses
    points = camera_relative_poses[:, 3:6]
    
    # Simple solid blue line for camera frame relative translations
    ax2.plot(points[:, 0], points[:, 1], points[:, 2], 
             c=blue_color, linewidth=0.5)
    
    # Add start and end points
    ax2.scatter(points[0, 0], points[0, 1], points[0, 2], 
               c='green', marker='o', s=20, label='Start')
    ax2.scatter(points[-1, 0], points[-1, 1], points[-1, 2], 
               c='red', marker='o', s=20, label='End')
    
    # Configure second plot
    ax2.set_xlabel('X (m)', labelpad=1)
    ax2.set_ylabel('Y (m)', labelpad=1)
    ax2.set_zlabel('Z (m)', labelpad=1)
    ax2.set_title('Camera Frame\nRelative Translations')
    ax2.grid(False)
    ax2.legend(loc='upper right', fontsize=6)
    
    # Set optimal view angles
    ax1.view_init(elev=30, azim=45)
    ax2.view_init(elev=30, azim=45)
    
    # Magnify the relative translations plot by using a smaller range
    max_val = np.max(np.abs(points))
    buffer = 0.1 * max_val  # 10% buffer
    
    # Set tighter limits for better visibility
    ax2.set_xlim([np.min(points[:, 0]) - buffer, np.max(points[:, 0]) + buffer])
    ax2.set_ylim([np.min(points[:, 1]) - buffer, np.max(points[:, 1]) + buffer])
    ax2.set_zlim([np.min(points[:, 2]) - buffer, np.max(points[:, 2]) + buffer])
    
    # Tight layout and save with high DPI for quality
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3)
    
    # Save the figure
    output_path = os.path.join(output_dir, 'ground_truth_vs_camera_frame.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print(f"Ground Truth Absolute vs Camera Frame Relative plot saved to {output_path}")
    return output_path

def analyze_ground_truth(pose_path):
    """
    Analyze ground truth poses to understand data ranges and distributions.
    
    Args:
        pose_path (str): Path to the pose file (.npy)
    """
    # Load poses
    if not os.path.exists(pose_path):
        print(f"Error: Pose file not found at {pose_path}")
        return None
    
    poses = np.load(pose_path)
    print(f"Loaded poses with shape: {poses.shape}")
    
    # Check format
    if poses.shape[1] != 6:
        print(f"Warning: Unexpected pose format. Expected 6 values per pose, got {poses.shape[1]}")
    
    # Analyze absolute pose ranges
    print("\n----- Absolute Pose Statistics -----")
    for i, label in enumerate(['Roll', 'Pitch', 'Yaw', 'X', 'Y', 'Z']):
        data = poses[:, i]
        print(f"{label}: min={data.min():.4f}, max={data.max():.4f}, mean={data.mean():.4f}, std={data.std():.4f}")
    
    # Compute and analyze relative poses
    relative_poses = []
    for i in range(1, len(poses)):
        rel_pose = compute_relative_pose(poses[i-1], poses[i])
        relative_poses.append(rel_pose)
    
    relative_poses = np.array(relative_poses)
    print("\n----- Relative Pose Statistics -----")
    for i, label in enumerate(['dRoll', 'dPitch', 'dYaw', 'dX', 'dY', 'dZ']):
        data = relative_poses[:, i]
        print(f"{label}: min={data.min():.4f}, max={data.max():.4f}, mean={data.mean():.4f}, std={data.std():.4f}")
    
    return poses, relative_poses

def compare_transformation_matrix(matrix, translation):
    """
    Display information about a specific transformation matrix
    
    Args:
        matrix (np.ndarray): 3x3 transformation matrix from body to camera frame
        translation (np.ndarray): 3D translation vector from body to camera (in meters)
    """
    print("\n----- Body to Camera Transformation -----")
    print("Rotation Matrix:")
    print(matrix)
    print("\nTranslation Vector:")
    print(translation)
    
    # Test vectors (unit vectors in body frame)
    test_vectors = np.eye(3)
    
    print("\nEffect on unit vectors:")
    for i, axis in enumerate(['X', 'Y', 'Z']):
        v = test_vectors[i]
        transformed = matrix @ v
        print(f"Body {axis}-axis → Camera: [{transformed[0]:.4f}, {transformed[1]:.4f}, {transformed[2]:.4f}]")
    
    # Visualize transformation
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot original frame
    origin = np.zeros(3)
    for i, (color, label) in enumerate(zip(['r', 'g', 'b'], ['X', 'Y', 'Z'])):
        ax.quiver(origin[0], origin[1], origin[2],
                 test_vectors[i, 0], test_vectors[i, 1], test_vectors[i, 2],
                 color=color, label=f'Body {label}')
    
    # Plot transformed frame
    for i, (color, label) in enumerate(zip(['r', 'g', 'b'], ['X', 'Y', 'Z'])):
        v = matrix @ test_vectors[i]
        ax.quiver(origin[0], origin[1], origin[2], v[0], v[1], v[2], 
                 color=color, alpha=0.5, linestyle='--', label=f'Camera {label}')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Body to Camera Transformation')
    ax.legend()
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_zlim([-1.1, 1.1])
    
    plt.tight_layout()
    plt.savefig('transformation_visualization.png', dpi=200)
    print(f"Transformation visualization saved as transformation_visualization.png")

def validate_midair_transformation(data_dir='/media/krkavinda/New Volume/Mid-Air-Dataset/MidAir_processed'):
    """
    Validate the MidAir coordinate transformation with your specific matrix.
    
    Args:
        data_dir (str): Base data directory for the MidAir dataset
    """
    output_dir = './transformation_validation'
    os.makedirs(output_dir, exist_ok=True)
    
    # Define trajectory ID
    trajectory = 'trajectory_0008'
    traj_number = trajectory.split('_')[1]
    
    # 1. Find the pose file
    climate_sets = [
        "Kite_training/cloudy", "Kite_training/foggy", "Kite_training/sunny", "Kite_training/sunset",
        "PLE_training/fall", "PLE_training/spring", "PLE_training/winter"
    ]
    
    pose_path = None
    
    for climate_set in climate_sets:
        path = os.path.join(data_dir, climate_set, "poses", f"poses_{traj_number}.npy")
        if os.path.exists(path):
            pose_path = path
            print(f"Found pose file: {pose_path}")
            break
    
    if pose_path is None:
        print(f"Error: Could not find pose file for trajectory {trajectory} in {data_dir}")
        # Use a path from the data_dir which should exist
        paths = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.startswith("poses_") and file.endswith(".npy"):
                    paths.append(os.path.join(root, file))
        
        if paths:
            pose_path = paths[0]
            print(f"Using alternative pose file: {pose_path}")
        else:
            print(f"Error: Could not find any pose files in {data_dir}")
            return
    
    # 2. Define your specific transformation matrix and translation offset
    body_to_camera = np.array([
        [0, 1, 0],
        [0, 0, -1],
        [-1, 0, 0]
    ])
    
    # Translation offset (from Figure 1: 1m North, 0m East, 0.5m Down)
    body_to_camera_translation = np.array([1.0, 0.0, 0.5])
    
    # 3. Analyze and display the transformation
    compare_transformation_matrix(body_to_camera, body_to_camera_translation)
    
    # 4. Analyze ground truth data
    poses, relative_poses = analyze_ground_truth(pose_path)
    
    if poses is None:
        return
    
    # 5. Validate transformation reversibility
    validate_transformation_reversibility(poses, body_to_camera, body_to_camera_translation)
    
    # 6. Generate trajectory visualization
    plot_ground_truth_absolute_vs_camera_relative(poses, body_to_camera, body_to_camera_translation, output_dir)
    
    # 7. Validate output value ranges
    print("\n----- Output Value Range Recommendations -----")
    # For angles
    print(f"Max absolute rotation: {np.max(np.abs(relative_poses[:, :3])):.4f} rad")
    # For translations
    print(f"Max absolute translation: {np.max(np.abs(relative_poses[:, 3:])):.4f} m")
    
    # Suggest scaling factors for model outputs
    max_rot = max(np.pi, np.max(np.abs(relative_poses[:, :3])))
    max_trans = max(5.0, np.max(np.abs(relative_poses[:, 3:])))
    
    print(f"\nRecommended output scaling factors:")
    print(f"max_rot = {max_rot:.4f} rad (use this in your model's forward method)")
    print(f"max_trans = {max_trans:.4f} m (use this in your model's forward method)")
    
    print(f"\nValidation complete. Results saved to {output_dir}")

if __name__ == "__main__":
    validate_midair_transformation()
