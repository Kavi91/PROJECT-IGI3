import h5py
import numpy as np
from scipy.spatial.transform import Rotation
import os
import glob
import re
from scipy.interpolate import interp1d
import pickle
import json
import logging

"""
Axis Convention:
- World Frame: NED (X=North, Y=East, Z=Down)
- Pose Format: [roll, pitch, yaw, x, y, z] (ZYX Euler angles, translations in meters)
- Output poses are normalized to start at [0, 0, 0, 0, 0, 0] in NED world frame
- Downsampled to 25Hz to match camera frame rate
"""

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
file_handler = logging.FileHandler('preprocessing.log')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Define paths - UPDATE THESE TO YOUR PATHS
base_path = '/media/krkavinda/New Volume/Mid-Air-Dataset/MidAir'
output_base = '/media/krkavinda/New Volume/Mid-Air-Dataset/MidAir_processed'

# Define climate sets
climate_sets = [
    'Kite_training/cloudy', 'Kite_training/foggy', 'Kite_training/sunny', 'Kite_training/sunset',
    'PLE_training/fall', 'PLE_training/spring', 'PLE_training/winter',
    'VO_test/foggy', 'VO_test/sunny', 'VO_test/sunset'
]

# Map trajectories to climate sets
traj_mapping = {
    'Kite_training/cloudy': [f'trajectory_{i:04d}' for i in range(3000, 3030)],
    'Kite_training/foggy': [f'trajectory_{i:04d}' for i in range(2000, 2030)],
    'Kite_training/sunny': [f'trajectory_{i:04d}' for i in range(0, 30)],
    'Kite_training/sunset': [f'trajectory_{i:04d}' for i in range(1000, 1030)],
    'PLE_training/fall': [f'trajectory_{i:04d}' for i in range(4000, 4024)],
    'PLE_training/spring': [f'trajectory_{i:04d}' for i in range(5000, 5024)],
    'PLE_training/winter': [f'trajectory_{i:04d}' for i in range(6000, 6024)],
    'VO_test/foggy': [f'trajectory_{i:04d}' for i in range(1000, 1003)],
    'VO_test/sunny': [f'trajectory_{i:04d}' for i in range(0, 3)],
    'VO_test/sunset': [f'trajectory_{i:04d}' for i in range(2000, 2003)]
}

# Calibration data (derived from Mid-Air documentation)
calib_content = (
    "P0: 512 0 512 0  0 384 384 0  0 0 1 0\n"
    "P1: 512 0 512 0.25  0 384 384 0  0 0 1 0\n"
)

# Define body-to-camera transformation
body_to_camera = np.array([
    [0, 1, 0],
    [0, 0, -1],
    [-1, 0, 0]
])
body_to_camera_translation = np.array([1.0, 0.0, 0.5])

# Dictionary to store frame counts for each trajectory
frame_counts = {}

# List to store trajectories with issues
skipped_trajectories = []
successful_trajectories = []

for climate_set in climate_sets:
    # Paths
    hdf5_path = f'{base_path}/{climate_set}/sensor_records.hdf5'
    output_dir = f'{output_base}/{climate_set}'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/poses', exist_ok=True)

    # Check if HDF5 file exists
    if not os.path.exists(hdf5_path):
        logger.error(f"HDF5 file not found for {climate_set}, skipping.")
        continue

    if climate_set not in frame_counts:
        frame_counts[climate_set] = {}

    # Process each trajectory
    for traj_key in traj_mapping.get(climate_set, []):
        logger.info(f"\nProcessing {climate_set}/{traj_key}")
        traj_success = True
        
        # Open HDF5 file for this trajectory
        try:
            with h5py.File(hdf5_path, 'r') as f:
                if traj_key not in f:
                    logger.error(f"Trajectory {traj_key} not found in {climate_set}, skipping.")
                    skipped_trajectories.append((climate_set, traj_key, "Trajectory not found in HDF5"))
                    continue
                traj = f[traj_key]

                # Symlink RGB (left, right, down) and depth for this trajectory
                traj_output_dir = f'{output_dir}/{traj_key}'
                os.makedirs(traj_output_dir, exist_ok=True)

                # Create symbolic links, skip if they exist
                link_rgb = f'{traj_output_dir}/image_rgb'
                link_rgb_right = f'{traj_output_dir}/image_rgb_right'
                link_rgb_down = f'{traj_output_dir}/image_rgb_down'
                link_depth = f'{traj_output_dir}/depth'

                # Create symbolic links only if the source directories exist
                # RGB Left
                src_rgb = f'{base_path}/{climate_set}/color_left/{traj_key}'
                if os.path.exists(src_rgb):
                    if not os.path.exists(link_rgb):
                        os.symlink(src_rgb, link_rgb)
                        logger.info(f"Created symbolic link for RGB Left: {link_rgb}")
                    else:
                        logger.info(f"Symbolic link {link_rgb} already exists.")
                else:
                    logger.warning(f"Source directory for RGB Left not found: {src_rgb}")
                    traj_success = False

                # RGB Right
                src_rgb_right = f'{base_path}/{climate_set}/color_right/{traj_key}'
                if os.path.exists(src_rgb_right):
                    if not os.path.exists(link_rgb_right):
                        os.symlink(src_rgb_right, link_rgb_right)
                        logger.info(f"Created symbolic link for RGB Right: {link_rgb_right}")
                    else:
                        logger.info(f"Symbolic link {link_rgb_right} already exists.")
                else:
                    logger.warning(f"Source directory for RGB Right not found: {src_rgb_right}")
                    traj_success = False

                # RGB Down
                src_rgb_down = f'{base_path}/{climate_set}/color_down/{traj_key}'
                if os.path.exists(src_rgb_down):
                    if not os.path.exists(link_rgb_down):
                        os.symlink(src_rgb_down, link_rgb_down)
                        logger.info(f"Created symbolic link for RGB Down: {link_rgb_down}")
                    else:
                        logger.info(f"Symbolic link {link_rgb_down} already exists.")
                else:
                    logger.warning(f"Source directory for RGB Down not found: {src_rgb_down}")
                    traj_success = False

                # Depth
                src_depth = f'{base_path}/{climate_set}/depth/{traj_key}'
                if os.path.exists(src_depth):
                    if not os.path.exists(link_depth):
                        os.symlink(src_depth, link_depth)
                        logger.info(f"Created symbolic link for Depth: {link_depth}")
                    else:
                        logger.info(f"Symbolic link {link_depth} already exists.")
                else:
                    logger.warning(f"Source directory for Depth not found: {src_depth}")
                    traj_success = False

                # Add calibration file
                calib_file_path = f'{traj_output_dir}/calib.txt'
                if not os.path.exists(calib_file_path):
                    with open(calib_file_path, 'w') as calib_file:
                        calib_file.write(calib_content)
                    logger.info(f"Created calibration file: {calib_file_path}")
                else:
                    logger.info(f"Calibration file {calib_file_path} already exists.")

                # Get the number of frames from the image files
                has_rgb_files = False
                num_rgb_left = 0
                valid_rgb_left_files = []
                
                if os.path.exists(src_rgb):
                    rgb_left_files = glob.glob(f'{src_rgb}/*.JPEG')
                    for f in rgb_left_files:
                        basename = os.path.basename(f).split('.')[0]
                        if re.match(r'^\d+$', basename):
                            valid_rgb_left_files.append(f)
                        else:
                            logger.warning(f"Skipping invalid filename: {f}")
                    
                    if valid_rgb_left_files:
                        has_rgb_files = True
                        valid_rgb_left_files = sorted(valid_rgb_left_files, 
                                                     key=lambda x: int(os.path.basename(x).split('.')[0]))
                        num_rgb_left = len(valid_rgb_left_files)

                # Get frame counts for other modalities if they exist
                num_rgb_right = 0
                if os.path.exists(src_rgb_right):
                    rgb_right_files = glob.glob(f'{src_rgb_right}/*.JPEG')
                    valid_files = [f for f in rgb_right_files if re.match(r'^\d+$', os.path.basename(f).split('.')[0])]
                    num_rgb_right = len(valid_files)

                num_rgb_down = 0
                if os.path.exists(src_rgb_down):
                    rgb_down_files = glob.glob(f'{src_rgb_down}/*.JPEG')
                    valid_files = [f for f in rgb_down_files if re.match(r'^\d+$', os.path.basename(f).split('.')[0])]
                    num_rgb_down = len(valid_files)

                num_depth = 0
                if os.path.exists(src_depth):
                    depth_files = glob.glob(f'{src_depth}/*.PNG')
                    valid_files = [f for f in depth_files if re.match(r'^\d+$', os.path.basename(f).split('.')[0])]
                    num_depth = len(valid_files)

                # Calculate the number of frames for cameras
                num_frames_camera = 0
                if has_rgb_files:
                    available_counts = [count for count in [num_rgb_left, num_rgb_right, num_rgb_down, num_depth] if count > 0]
                    if available_counts:
                        num_frames_camera = min(available_counts)
                        logger.info(f"{climate_set}/{traj_key}: Camera frames (RGB Left={num_rgb_left}, RGB Right={num_rgb_right}, RGB Down={num_rgb_down}, Depth={num_depth})")
                    else:
                        logger.warning(f"No valid camera frames found for {climate_set}/{traj_key}")
                        traj_success = False
                else:
                    logger.warning(f"No RGB files found for {climate_set}/{traj_key}")
                    traj_success = False

                # Get ground truth poses at 25Hz
                if 'groundtruth' not in traj:
                    logger.error(f"Groundtruth data missing for {climate_set}/{traj_key}, skipping trajectory.")
                    skipped_trajectories.append((climate_set, traj_key, "Missing groundtruth data"))
                    continue

                # Get positions and attitudes, downsample to 25Hz
                gt_positions = np.array(traj['groundtruth']['position'])
                gt_attitudes = np.array(traj['groundtruth']['attitude'])
                
                # Check if we need to downsample (assuming 100Hz -> 25Hz)
                downsample_factor = 4
                frame_rate = traj['groundtruth'].attrs.get('frame_rate', 100)  # Default to 100Hz if not specified
                if frame_rate != 100:
                    # Adjust downsample factor for different source frame rates
                    downsample_factor = int(frame_rate / 25)
                    logger.info(f"Adjusted downsample factor to {downsample_factor} for frame rate {frame_rate}Hz")
                
                positions = gt_positions[::downsample_factor]  # Downsample to 25Hz
                attitudes = gt_attitudes[::downsample_factor]  # Downsample to 25Hz
                num_poses = len(positions)
                
                logger.info(f"{climate_set}/{traj_key}: Number of ground truth poses after downsampling = {num_poses}")

                # Determine the final number of frames to use
                if num_frames_camera > 0 and num_poses > 0:
                    num_frames = min(num_frames_camera, num_poses)
                    logger.info(f"Using {num_frames} frames for {climate_set}/{traj_key}")
                    
                    # Truncate camera frames and poses to match each other
                    if num_frames_camera != num_poses:
                        logger.warning(f"Camera frame count ({num_frames_camera}) does not match ground truth ({num_poses}), adjusting to {num_frames} frames.")
                        if has_rgb_files:
                            valid_rgb_left_files = valid_rgb_left_files[:num_frames]
                        positions = positions[:num_frames]
                        attitudes = attitudes[:num_frames]
                else:
                    if num_frames_camera == 0:
                        logger.error(f"No valid camera frames for {climate_set}/{traj_key}, skipping.")
                        skipped_trajectories.append((climate_set, traj_key, "No valid camera frames"))
                        continue
                    if num_poses == 0:
                        logger.error(f"No valid poses for {climate_set}/{traj_key}, skipping.")
                        skipped_trajectories.append((climate_set, traj_key, "No valid poses"))
                        continue

                # Validate image indices for synchronization if RGB files exist
                if has_rgb_files:
                    actual_indices = [int(os.path.basename(f).split('.')[0]) for f in valid_rgb_left_files]
                    expected_indices = list(range(len(valid_rgb_left_files)))
                    missing_indices = set(expected_indices) - set(actual_indices)
                    if missing_indices:
                        logger.warning(f"Missing frames {missing_indices} in {climate_set}/{traj_key}. Proceeding with available frames.")
                        # Adjust num_frames if needed
                        num_frames = min(num_frames, len(actual_indices))
                        valid_rgb_left_files = valid_rgb_left_files[:num_frames]
                        positions = positions[:num_frames]
                        attitudes = attitudes[:num_frames]

                # Convert quaternions to Euler angles (ZYX order for NED)
                rot = Rotation.from_quat(attitudes[:, [1, 2, 3, 0]])  # [qx, qy, qz, qw]
                euler = rot.as_euler('zyx', degrees=False)  # [yaw, pitch, roll]
                euler = euler[:, [2, 1, 0]]  # Reorder to [roll, pitch, yaw]

                # Normalize the initial orientation and position
                initial_euler = euler[0].copy()
                euler -= initial_euler
                initial_position = positions[0].copy()
                positions -= initial_position

                # Save poses in Body frame (NED) as [roll, pitch, yaw, x, y, z]
                body_poses = np.hstack((euler, positions))

                # Log raw positions and relative translations
                logger.info(f"{climate_set}/{traj_key}: First few raw positions (Body frame): {positions[:5]}")
                raw_deltas = np.diff(positions, axis=0)
                raw_max_translation = np.max(np.linalg.norm(raw_deltas, axis=1)) if len(raw_deltas) > 0 else 0
                logger.info(f"{climate_set}/{traj_key}: Max raw translation (Body frame): {raw_max_translation}")

                # Print the first pose to verify alignment
                logger.info(f"{climate_set}/{traj_key}: First body frame pose (roll, pitch, yaw, x, y, z): {body_poses[0]}")

                # Save body frame poses
                traj_id = traj_key.split("_")[1]
                body_pose_file = f'{output_dir}/poses/poses_{traj_id}.npy'
                np.save(body_pose_file, body_poses)
                logger.info(f"Saved body frame poses for {traj_key}, shape: {body_poses.shape}")
                
                # Transform poses to camera frame
                camera_poses = np.zeros_like(body_poses)
                
                for i in range(len(body_poses)):
                    # Extract rotation and translation from body frame
                    body_euler = body_poses[i, :3]
                    body_position = body_poses[i, 3:6]
                    
                    # Convert Euler angles to rotation matrix (in body frame)
                    # We need to convert from [roll, pitch, yaw] to [yaw, pitch, roll] for scipy's from_euler
                    body_R = Rotation.from_euler('zyx', body_euler[::-1]).as_matrix()
                    
                    # Transform rotation to camera frame
                    camera_R = body_to_camera @ body_R @ np.linalg.inv(body_to_camera)
                    
                    # Convert camera rotation matrix back to Euler angles
                    # Convert back from [yaw, pitch, roll] to [roll, pitch, yaw]
                    camera_euler = Rotation.from_matrix(camera_R).as_euler('zyx')[::-1]
                    
                    # Transform position to camera frame
                    camera_position = body_to_camera @ body_position + body_to_camera_translation
                    
                    # Store camera frame pose
                    camera_poses[i, :3] = camera_euler
                    camera_poses[i, 3:6] = camera_position
                
                # Log first camera frame pose
                logger.info(f"{climate_set}/{traj_key}: First camera frame pose (roll, pitch, yaw, x, y, z): {camera_poses[0]}")
                
                # Save camera frame poses
                camera_pose_file = f'{output_dir}/poses/camera_poses_{traj_id}.npy'
                np.save(camera_pose_file, camera_poses)
                logger.info(f"Saved camera frame poses for {traj_key}, shape: {camera_poses.shape}")
                
                # Check if files were created successfully
                if not os.path.exists(body_pose_file) or not os.path.exists(camera_pose_file):
                    logger.error(f"Error: Pose file(s) were not created successfully.")
                    skipped_trajectories.append((climate_set, traj_key, "Failed to save pose file"))
                    continue

                # Save the adjusted frame count
                frame_counts[climate_set][traj_id] = num_frames

                # IMPROVED IMU PROCESSING SECTION
                imu_success = True
                try:
                    # Check if IMU data is available
                    if 'imu' not in traj:
                        logger.warning(f"IMU data missing for {climate_set}/{traj_key}. Will proceed without IMU.")
                        imu_success = False
                    else:
                        imu_group = traj['imu']
                        
                        # Get IMU metadata
                        imu_frame_rate = imu_group.attrs.get('frame_rate', 100)  # Default to 100Hz if not specified
                        imu_downsample_factor = int(imu_frame_rate / 25)  # Calculate factor to get 25Hz
                        
                        # Check if required data exists
                        if 'accelerometer' not in imu_group or 'gyroscope' not in imu_group:
                            logger.warning(f"IMU accelerometer or gyroscope missing for {climate_set}/{traj_key}. Will proceed without IMU.")
                            imu_success = False
                        else:
                            # Get accelerometer and gyroscope data
                            raw_accelerometer = np.array(imu_group['accelerometer'])
                            raw_gyroscope = np.array(imu_group['gyroscope'])
                            
                            # Get initial bias values, handle different formats
                            try:
                                init_bias_acc = np.array(imu_group['accelerometer'].attrs.get('init_bias_est', np.zeros(3)))
                                init_bias_gyr = np.array(imu_group['gyroscope'].attrs.get('init_bias_est', np.zeros(3)))
                                
                                # Ensure correct shape
                                if init_bias_acc.ndim > 1:
                                    init_bias_acc = init_bias_acc.flatten()[:3]
                                if init_bias_gyr.ndim > 1:
                                    init_bias_gyr = init_bias_gyr.flatten()[:3]
                                
                                # If bias data is not available, use zeros
                                if len(init_bias_acc) != 3:
                                    init_bias_acc = np.zeros(3)
                                if len(init_bias_gyr) != 3:
                                    init_bias_gyr = np.zeros(3)
                                    
                                init_bias = np.vstack((init_bias_acc, init_bias_gyr))  # Shape: [2, 3]
                            except Exception as e:
                                logger.warning(f"Error extracting IMU bias for {climate_set}/{traj_key}: {str(e)}. Using zeros.")
                                init_bias = np.zeros((2, 3))
                            
                            # Downsample IMU data to match pose frequency (25Hz)
                            accelerometer = raw_accelerometer[::imu_downsample_factor]
                            gyroscope = raw_gyroscope[::imu_downsample_factor]
                            
                            # Ensure IMU data is aligned with the number of frames
                            acc_len = len(accelerometer)
                            gyr_len = len(gyroscope)
                            
                            # Use the smaller of the two and truncate to match num_frames
                            min_imu_len = min(acc_len, gyr_len)
                            if min_imu_len < num_frames:
                                logger.warning(f"IMU data shorter than num_frames: {min_imu_len} vs {num_frames}. Padding.")
                                # Pad with the last value if shorter
                                if acc_len < num_frames:
                                    pad_width = ((0, num_frames - acc_len), (0, 0))
                                    accelerometer = np.pad(accelerometer, pad_width, mode='edge')
                                if gyr_len < num_frames:
                                    pad_width = ((0, num_frames - gyr_len), (0, 0))
                                    gyroscope = np.pad(gyroscope, pad_width, mode='edge')
                            else:
                                # Truncate to match num_frames
                                accelerometer = accelerometer[:num_frames]
                                gyroscope = gyroscope[:num_frames]
                            
                            # Combine accelerometer and gyroscope data (in body frame)
                            body_imu_data = np.hstack((accelerometer, gyroscope))  # Shape: [num_frames, 6]
                            
                            # Transform IMU data to camera frame
                            camera_imu_data = np.zeros_like(body_imu_data)
                            
                            for i in range(len(body_imu_data)):
                                # Transform accelerometer readings to camera frame
                                acc_body = body_imu_data[i, 0:3]
                                acc_camera = body_to_camera @ acc_body
                                
                                # Transform gyroscope readings to camera frame
                                gyro_body = body_imu_data[i, 3:6]
                                gyro_camera = body_to_camera @ gyro_body
                                
                                # Store transformed IMU data
                                camera_imu_data[i, 0:3] = acc_camera
                                camera_imu_data[i, 3:6] = gyro_camera
                            
                            # Save both body frame and camera frame IMU data
                            imu_output_dir = f'{traj_output_dir}/imu'
                            os.makedirs(imu_output_dir, exist_ok=True)
                            
                            # Save body frame IMU data
                            np.save(f'{imu_output_dir}/imu.npy', body_imu_data)
                            np.save(f'{imu_output_dir}/imu_init_bias.npy', init_bias)
                            logger.info(f"Saved body frame IMU data for {traj_key}, shape: {body_imu_data.shape}")
                            
                            # Save camera frame IMU data
                            np.save(f'{imu_output_dir}/camera_imu.npy', camera_imu_data)
                            logger.info(f"Saved camera frame IMU data for {traj_key}, shape: {camera_imu_data.shape}")
                except Exception as e:
                    logger.error(f"Error processing IMU data for {climate_set}/{traj_key}: {str(e)}")
                    imu_success = False

                # IMPROVED GPS PROCESSING SECTION
                gps_success = True
                try:
                    # Check if GPS data is available
                    if 'gps' not in traj:
                        logger.warning(f"GPS data missing for {climate_set}/{traj_key}. Will proceed without GPS.")
                        gps_success = False
                    else:
                        gps_group = traj['gps']
                        
                        # Check required GPS datasets
                        required_gps_datasets = ['position', 'velocity']
                        optional_gps_datasets = ['no_vis_sats', 'GDOP', 'PDOP', 'HDOP', 'VDOP']
                        
                        missing_required = [ds for ds in required_gps_datasets if ds not in gps_group]
                        if missing_required:
                            logger.warning(f"Missing required GPS datasets {missing_required} for {climate_set}/{traj_key}. Will proceed without GPS.")
                            gps_success = False
                        else:
                            # Get position and velocity data
                            gps_position = np.array(gps_group['position'])
                            gps_velocity = np.array(gps_group['velocity'])
                            num_gps_frames = len(gps_position)
                            
                            # Create signal info array (will be filled with available data or zeros)
                            gps_signal = np.zeros((num_gps_frames, 5))
                            
                            # Fill signal info with available data
                            for i, dataset in enumerate(optional_gps_datasets):
                                if dataset in gps_group and len(gps_group[dataset]) == num_gps_frames:
                                    try:
                                        data = np.array(gps_group[dataset])
                                        if data.ndim > 1 and data.shape[1] > 0:
                                            gps_signal[:, i] = data[:, 0]
                                        elif data.ndim == 1:
                                            gps_signal[:, i] = data
                                    except Exception as e:
                                        logger.warning(f"Error processing GPS {dataset} for {climate_set}/{traj_key}: {str(e)}")
                            
                            # Handle NaN values in GPS data
                            if np.any(np.isnan(gps_position)) or np.any(np.isnan(gps_velocity)) or np.any(np.isnan(gps_signal)):
                                logger.warning(f"GPS data contains NaN values for {climate_set}/{traj_key}. Replacing with interpolation or zeros.")
                                
                                # Replace NaNs in position with interpolated values or zeros
                                for i in range(gps_position.shape[1]):
                                    mask = np.isnan(gps_position[:, i])
                                    if np.all(mask):
                                        gps_position[:, i] = 0
                                    elif np.any(mask):
                                        valid_indices = np.where(~mask)[0]
                                        valid_values = gps_position[valid_indices, i]
                                        
                                        # Create interpolation function for valid values
                                        if len(valid_indices) > 1:
                                            interp_func = interp1d(valid_indices, valid_values, 
                                                                 bounds_error=False, fill_value="extrapolate")
                                            gps_position[:, i] = interp_func(np.arange(len(gps_position)))
                                        else:
                                            # If only one valid value, fill with it
                                            gps_position[:, i] = valid_values[0] if len(valid_values) > 0 else 0
                                
                                # Do the same for velocity
                                for i in range(gps_velocity.shape[1]):
                                    mask = np.isnan(gps_velocity[:, i])
                                    if np.all(mask):
                                        gps_velocity[:, i] = 0
                                    elif np.any(mask):
                                        valid_indices = np.where(~mask)[0]
                                        valid_values = gps_velocity[valid_indices, i]
                                        
                                        if len(valid_indices) > 1:
                                            interp_func = interp1d(valid_indices, valid_values, 
                                                                 bounds_error=False, fill_value="extrapolate")
                                            gps_velocity[:, i] = interp_func(np.arange(len(gps_velocity)))
                                        else:
                                            gps_velocity[:, i] = valid_values[0] if len(valid_values) > 0 else 0
                                
                                # For signal data, replace NaNs with zeros
                                gps_signal = np.nan_to_num(gps_signal)
                            
                            # Interpolate GPS to match num_frames (from 1Hz to 25Hz)
                            gps_times = np.arange(num_gps_frames)
                            target_times = np.linspace(0, num_gps_frames - 1, num_frames)
                            
                            # Create interpolation functions for position and velocity
                            interp_pos = interp1d(gps_times, gps_position, axis=0, kind='linear', 
                                                 fill_value="extrapolate")
                            interp_vel = interp1d(gps_times, gps_velocity, axis=0, kind='linear',
                                                 fill_value="extrapolate")
                            
                            # Interpolate position and velocity to match target frames
                            gps_position_interp = interp_pos(target_times)
                            gps_velocity_interp = interp_vel(target_times)
                            
                            # For signal data, use nearest neighbor interpolation
                            signal_indices = np.round(target_times).astype(int)
                            signal_indices = np.clip(signal_indices, 0, num_gps_frames - 1)
                            gps_signal_interp = gps_signal[signal_indices]
                            
                            # Combine position, velocity, and signal info for body frame
                            body_gps_data = np.hstack((
                                gps_position_interp,                  # Position [x, y, z]
                                gps_velocity_interp,                  # Velocity [vx, vy, vz]
                                gps_signal_interp[:, :3]              # Signal info [num_sats, GDOP, PDOP]
                            ))
                            
                            # Transform GPS position and velocity to camera frame
                            camera_gps_data = np.zeros_like(body_gps_data)
                            
                            for i in range(len(body_gps_data)):
                                # Transform position to camera frame
                                pos_body = body_gps_data[i, 0:3]
                                pos_camera = body_to_camera @ pos_body + body_to_camera_translation
                                
                                # Transform velocity to camera frame
                                vel_body = body_gps_data[i, 3:6]
                                vel_camera = body_to_camera @ vel_body
                                
                                # Copy signal info unchanged
                                signal_info = body_gps_data[i, 6:9]
                                
                                # Store transformed GPS data
                                camera_gps_data[i, 0:3] = pos_camera
                                camera_gps_data[i, 3:6] = vel_camera
                                camera_gps_data[i, 6:9] = signal_info
                            
                            # Save both body frame and camera frame GPS data
                            gps_output_dir = f'{traj_output_dir}/gps'
                            os.makedirs(gps_output_dir, exist_ok=True)
                            
                            # Save body frame GPS data
                            np.save(f'{gps_output_dir}/gps.npy', body_gps_data)
                            logger.info(f"Saved body frame GPS data for {traj_key}, shape: {body_gps_data.shape}")
                            
                            # Save camera frame GPS data
                            np.save(f'{gps_output_dir}/camera_gps.npy', camera_gps_data)
                            logger.info(f"Saved camera frame GPS data for {traj_key}, shape: {camera_gps_data.shape}")
                except Exception as e:
                    logger.error(f"Error processing GPS data for {climate_set}/{traj_key}: {str(e)}")
                    gps_success = False

                # Consider trajectory successful if poses were saved, even if IMU/GPS failed
                if traj_success:
                    successful_trajectories.append(f"{climate_set}/{traj_key}")
                    logger.info(f"Successfully processed {climate_set}/{traj_key}")
                else:
                    skipped_trajectories.append((climate_set, traj_key, "Processing issues"))
                    logger.warning(f"Had issues processing {climate_set}/{traj_key}, but continuing")

        except Exception as e:
            logger.error(f"Error processing {climate_set}/{traj_key}: {str(e)}")
            skipped_trajectories.append((climate_set, traj_key, f"Processing error: {str(e)}"))
            continue

    # Save frame counts for this climate set
    with open(f'{output_dir}/frame_counts.json', 'w') as f:
        json.dump(frame_counts[climate_set], f)
    logger.info(f"Saved frame counts for {climate_set} to {output_dir}/frame_counts.json")

# Save the full frame counts dictionary
with open(f'{output_base}/frame_counts.json', 'w') as f:
    json.dump(frame_counts, f)
logger.info(f"\nSaved all frame counts to {output_base}/frame_counts.json")

# Summary of processing
logger.info("\n=== Processing Summary ===")
logger.info(f"Total trajectories processed: {len(successful_trajectories) + len(skipped_trajectories)}")
logger.info(f"Successfully processed: {len(successful_trajectories)}")
logger.info(f"Skipped trajectories: {len(skipped_trajectories)}")

# Summary of skipped trajectories
if skipped_trajectories:
    logger.info("\n=== Summary of Skipped Trajectories ===")
    for climate_set, traj_key, reason in skipped_trajectories:
        logger.info(f"Skipped {climate_set}/{traj_key}: {reason}")

# Save successful trajectory mapping
successful_traj_mapping = {}
for climate_set in climate_sets:
    successful_traj_mapping[climate_set] = []
    for traj in successful_trajectories:
        if traj.startswith(climate_set + '/'):  # Ensure exact match with trailing slash
            traj_key = traj[len(climate_set) + 1:]  # Extract trajectory key (e.g., trajectory_XXXX)
            successful_traj_mapping[climate_set].append(traj_key)

# Log the successful trajectory mapping for verification
logger.info("\nSuccessful trajectory mapping:")
for climate_set, trajs in successful_traj_mapping.items():
    logger.info(f"  {climate_set}: {len(trajs)} trajectories")
    logger.debug(f"    Trajectories: {trajs}")

# Save updated traj_mapping
with open('successful_traj_mapping.pickle', 'wb') as f:
    pickle.dump(successful_traj_mapping, f)
logger.info("\nSaved successful trajectory mapping to successful_traj_mapping.pickle")