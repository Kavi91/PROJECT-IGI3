import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class IMUPreprocessor:
    """
    Preprocesses IMU data by applying bias correction and integration.
    Handles IMU data already downsampled to 25Hz to match camera frames.
    """
    def __init__(self, 
                 gyro_bias_init=None, 
                 acc_bias_init=None,
                 apply_integration=True,
                 dt=0.04,  # 25Hz (1/25 = 0.04s)
                 gravity=9.81,
                 body_to_camera=None):
        """
        Args:
            gyro_bias_init: Initial gyroscope bias estimate [3]
            acc_bias_init: Initial accelerometer bias estimate [3]
            apply_integration: Whether to integrate IMU measurements
            dt: Time step between IMU measurements in seconds (now 0.04 for 25Hz)
            gravity: Gravity constant
            body_to_camera: 3x3 rotation matrix for body-to-camera transformation
        """
        self.gyro_bias = gyro_bias_init if gyro_bias_init is not None else np.zeros(3)
        self.acc_bias = acc_bias_init if acc_bias_init is not None else np.zeros(3)
        self.apply_integration = apply_integration
        self.dt = dt
        self.gravity = gravity
        
        # Body to camera transformation (default is identity)
        self.body_to_camera = body_to_camera if body_to_camera is not None else np.eye(3)
        self.camera_to_body = np.linalg.inv(self.body_to_camera)
        
        # Gravity vector in world frame (assuming NED: North-East-Down)
        self.g_w = np.array([0, 0, gravity])
    
    def correct_bias(self, imu_data):
        """
        Apply bias correction to IMU data.
        
        Args:
            imu_data: IMU data with shape [seq_len, 6] where 
                      [:, 0:3] is accelerometer and [:, 3:6] is gyroscope
        
        Returns:
            Corrected IMU data with shape [seq_len, 6]
        """
        # Make a copy to avoid modifying the input
        corrected = imu_data.copy()
        
        # Apply bias correction
        corrected[:, 0:3] -= self.acc_bias
        corrected[:, 3:6] -= self.gyro_bias
        
        # Transform from body to camera frame if needed (only if not identity)
        if not np.allclose(self.body_to_camera, np.eye(3)):
            for i in range(corrected.shape[0]):
                # Transform accelerometer data
                corrected[i, 0:3] = self.body_to_camera @ corrected[i, 0:3]
                # Transform gyroscope data
                corrected[i, 3:6] = self.body_to_camera @ corrected[i, 3:6]
        
        return corrected
    
    def integrate_imu(self, imu_data, initial_orientation=None, initial_velocity=None, initial_position=None):
        """
        Integrate IMU measurements to get orientation, velocity, and position.
        Note: This function is for short-term integration between frames.
        
        Args:
            imu_data: Bias-corrected IMU data [seq_len, 6]
            initial_orientation: Initial orientation as rotation matrix [3, 3]
            initial_velocity: Initial velocity [3]
            initial_position: Initial position [3]
            
        Returns:
            orientations: Array of rotation matrices [seq_len, 3, 3]
            velocities: Array of velocities [seq_len, 3]
            positions: Array of positions [seq_len, 3]
        """
        seq_len = imu_data.shape[0]
        
        # Initialize arrays
        orientations = np.zeros((seq_len, 3, 3))
        velocities = np.zeros((seq_len, 3))
        positions = np.zeros((seq_len, 3))
        
        # Set initial values
        if initial_orientation is None:
            orientations[0] = np.eye(3)  # Identity rotation
        else:
            orientations[0] = initial_orientation
            
        if initial_velocity is not None:
            velocities[0] = initial_velocity
            
        if initial_position is not None:
            positions[0] = initial_position
        
        # Integrate
        for i in range(1, seq_len):
            # Get IMU measurements
            acc = imu_data[i-1, 0:3]
            gyro = imu_data[i-1, 3:6]
            
            # Update orientation using gyroscope (simple Euler integration)
            # Convert angular velocity to rotation matrix
            angle = np.linalg.norm(gyro)
            if angle > 1e-10:
                axis = gyro / angle
                skew = np.array([[0, -axis[2], axis[1]],
                                 [axis[2], 0, -axis[0]],
                                 [-axis[1], axis[0], 0]])
                rot_delta = np.eye(3) + np.sin(angle * self.dt) * skew + \
                            (1 - np.cos(angle * self.dt)) * (skew @ skew)
                orientations[i] = orientations[i-1] @ rot_delta
            else:
                orientations[i] = orientations[i-1]
            
            # Remove gravity and transform acceleration to world frame
            acc_world = orientations[i] @ acc - self.g_w
            
            # Update velocity and position (Euler integration)
            velocities[i] = velocities[i-1] + acc_world * self.dt
            positions[i] = positions[i-1] + velocities[i-1] * self.dt + 0.5 * acc_world * (self.dt ** 2)
        
        return orientations, velocities, positions
    
    def process_sequence(self, imu_data, integrate_segments=True, seq_len=None):
        """
        Process a sequence of IMU data for visual-inertial odometry.
        IMU data is already at 25Hz, matching camera frames.
        
        Args:
            imu_data: Raw IMU data [seq_len, 6]
            integrate_segments: Whether to integrate IMU data between frames
            seq_len: Required output sequence length (if None, determine from data)
            
        Returns:
            Processed IMU features for each camera frame [n_frames, feature_dim]
        """
        if imu_data is None or len(imu_data) < 1:
            return None
            
        # Apply bias correction
        corrected_imu = self.correct_bias(imu_data)
        
        # Determine number of frames
        n_frames = len(corrected_imu)
        
        # If we have fewer frames than required sequence length, or sequence length not specified
        if seq_len is None:
            # Use all available frames
            required_frames = n_frames
        else:
            # Use the required sequence length
            required_frames = seq_len
        
        # Ensure we have at least one frame
        if required_frames < 1:
            required_frames = 1
        
        # Prepare output array - size depends on feature type
        feature_dim = 21 if integrate_segments and self.apply_integration else 6
        imu_features = np.zeros((required_frames, feature_dim))
        
        # Process each frame
        for i in range(min(n_frames, required_frames)):
            if integrate_segments and self.apply_integration:
                # Use the current and next frame for integration (if available)
                end_idx = min(i + 2, len(corrected_imu))
                segment = corrected_imu[i:end_idx]
                
                try:
                    orientations, velocities, positions = self.integrate_imu(segment)
                    
                    # Extract features:
                    # - Mean accelerometer and gyroscope
                    # - Final estimated orientation change (as flattened matrix difference from identity)
                    # - Final velocity
                    # - Total displacement
                    mean_acc = np.mean(segment[:, 0:3], axis=0)
                    mean_gyro = np.mean(segment[:, 3:6], axis=0)
                    
                    if len(orientations) > 1:  # Need at least 2 frames for orientation change
                        # Compute orientation change from start to end
                        orientation_change = orientations[-1] - np.eye(3)
                        orientation_change = orientation_change.flatten()
                        final_velocity = velocities[-1] if len(velocities) > 0 else np.zeros(3)
                        total_displacement = positions[-1] if len(positions) > 0 else np.zeros(3)
                    else:
                        # If only one frame, use zeros for derived features
                        orientation_change = np.zeros(9)
                        final_velocity = np.zeros(3)
                        total_displacement = np.zeros(3)
                    
                    features = np.concatenate([
                        mean_acc,               # 3D
                        mean_gyro,              # 3D
                        orientation_change,     # 9D
                        final_velocity,         # 3D
                        total_displacement      # 3D
                    ])
                    imu_features[i] = features
                except Exception as e:
                    print(f"IMU integration error: {e}")
                    mean_acc = np.mean(segment[:, 0:3], axis=0) if len(segment) > 0 else np.zeros(3)
                    mean_gyro = np.mean(segment[:, 3:6], axis=0) if len(segment) > 0 else np.zeros(3)
                    zeros = np.zeros(15)
                    imu_features[i] = np.concatenate([mean_acc, mean_gyro, zeros])
            else:
                # Simple features: just use the IMU sample at this frame
                imu_features[i] = corrected_imu[i]
        
        # Fill remaining required frames with repeats or zeros
        if n_frames < required_frames:
            # Use last valid frame data or zeros
            if n_frames > 0:
                # Repeat last frame data
                for i in range(n_frames, required_frames):
                    imu_features[i] = imu_features[n_frames-1]
            else:
                # All zeros if no valid frames
                pass  # Already initialized with zeros
        
        return imu_features

class IMUFeatureExtractor(nn.Module):
    """
    Extracts features from IMU data for visual-inertial odometry.
    """
    def __init__(self, input_size=21, hidden_size=128, output_size=256, dropout=0.2):
        """
        Args:
            input_size: Number of IMU features (21 for integrated features, 6 for raw)
            hidden_size: Size of hidden layers
            output_size: Size of output features
            dropout: Dropout probability
        """
        super(IMUFeatureExtractor, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Feature extraction network
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
            nn.LeakyReLU(0.1)
        )
        
    def forward(self, imu_features):
        """
        Process IMU features.
        
        Args:
            imu_features: Tensor of shape [batch_size, seq_len, input_size]
        
        Returns:
            IMU features of shape [batch_size, seq_len, output_size]
        """
        batch_size, seq_len, _ = imu_features.shape
        
        # Reshape for batch processing
        x = imu_features.view(-1, self.input_size)
        
        # Extract features
        x = self.encoder(x)
        
        # Reshape back
        x = x.view(batch_size, seq_len, self.output_size)
        
        return x

class AdaptiveWeighting(nn.Module):
    """
    Adaptive weighting mechanism for visual and inertial features.
    Learns to weight features based on their reliability.
    """
    def __init__(self, visual_size, imu_size, output_size):
        """
        Args:
            visual_size: Size of visual features
            imu_size: Size of IMU features
            output_size: Size of output features
        """
        super(AdaptiveWeighting, self).__init__()
        
        self.visual_size = visual_size
        self.imu_size = imu_size
        self.output_size = output_size
        
        # Network to predict weights for visual and IMU features
        self.weighting_network = nn.Sequential(
            nn.Linear(visual_size + imu_size, 128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(128, 2),  # Two weights: one for visual, one for IMU
            nn.Softmax(dim=-1)  # Ensure weights sum to 1
        )
        
        # Projection layers for visual and IMU features
        self.visual_proj = nn.Linear(visual_size, output_size)
        self.imu_proj = nn.Linear(imu_size, output_size)
        
    def forward(self, visual_features, imu_features):
        """
        Adaptively weight and combine visual and IMU features.
        
        Args:
            visual_features: Visual features [batch_size, seq_len, visual_size]
            imu_features: IMU features [batch_size, seq_len, imu_size]
        
        Returns:
            Weighted and combined features [batch_size, seq_len, output_size]
        """
        # Combine features to predict weights
        combined = torch.cat([visual_features, imu_features], dim=-1)
        weights = self.weighting_network(combined)  # [batch_size, seq_len, 2]
        
        # Get weights for each modality
        visual_weight = weights[:, :, 0:1]  # [batch_size, seq_len, 1]
        imu_weight = weights[:, :, 1:2]     # [batch_size, seq_len, 1]
        
        # Project features to common space
        visual_proj = self.visual_proj(visual_features)
        imu_proj = self.imu_proj(imu_features)
        
        # Apply weights and combine
        weighted_visual = visual_proj * visual_weight
        weighted_imu = imu_proj * imu_weight
        
        # Sum weighted features
        fused = weighted_visual + weighted_imu
        
        return fused

class VisualInertialFusion(nn.Module):
    """
    Fuses visual and inertial features for visual-inertial odometry.
    """
    def __init__(self, visual_size, imu_size, fusion_size=512, use_adaptive_weighting=True):
        """
        Args:
            visual_size: Size of visual features
            imu_size: Size of IMU features
            fusion_size: Size of fused features
            use_adaptive_weighting: Whether to use adaptive weighting
        """
        super(VisualInertialFusion, self).__init__()
        
        self.use_adaptive_weighting = use_adaptive_weighting
        
        if use_adaptive_weighting:
            # Adaptive weighting fusion
            self.fusion = AdaptiveWeighting(
                visual_size=visual_size,
                imu_size=imu_size,
                output_size=fusion_size
            )
        else:
            # Simple concatenation and projection
            self.visual_proj = nn.Linear(visual_size, fusion_size // 2)
            self.imu_proj = nn.Linear(imu_size, fusion_size // 2)
            self.fusion_layer = nn.Sequential(
                nn.Linear(fusion_size, fusion_size),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.2)
            )
        
    def forward(self, visual_features, imu_features):
        """
        Fuse visual and IMU features.
        
        Args:
            visual_features: Visual features [batch_size, seq_len, visual_size]
            imu_features: IMU features [batch_size, seq_len, imu_size]
        
        Returns:
            Fused features [batch_size, seq_len, fusion_size]
        """
        if self.use_adaptive_weighting:
            # Use adaptive weighting
            fused = self.fusion(visual_features, imu_features)
        else:
            # Project to common size
            visual_proj = self.visual_proj(visual_features)
            imu_proj = self.imu_proj(imu_features)
            
            # Concatenate and fuse
            concat = torch.cat([visual_proj, imu_proj], dim=-1)
            fused = self.fusion_layer(concat)
        
        return fused