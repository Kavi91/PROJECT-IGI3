import torch
import torch.nn.functional as F
import torchvision.transforms as T
import random
import numpy as np
from PIL import Image

class SequenceAugmenter:
    """
    Applies consistent augmentations across an entire sequence of frames.
    This is crucial for visual odometry where temporal/geometric consistency must be preserved.
    """
    def __init__(self, 
                 color_jitter_prob=0.8,
                 brightness_range=0.3, 
                 contrast_range=0.3, 
                 saturation_range=0.3,
                 hue_range=0.1,
                 noise_prob=0.3,
                 noise_std=0.02):
        """
        Args:
            color_jitter_prob: Probability of applying color jitter to the sequence
            brightness_range: Max brightness shift (±value)
            contrast_range: Max contrast shift (±value)
            saturation_range: Max saturation shift (±value)
            hue_range: Max hue shift (±value)
            noise_prob: Probability of applying noise to the sequence
            noise_std: Standard deviation of Gaussian noise
        """
        self.color_jitter_prob = color_jitter_prob
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_range = hue_range
        self.noise_prob = noise_prob
        self.noise_std = noise_std
        
    def __call__(self, seq_tensor):
        """
        Apply augmentations consistently to a sequence of frames
        
        Args:
            seq_tensor: Sequence of frames tensor [seq_len, channels, height, width]
            
        Returns:
            Augmented sequence tensor [seq_len, channels, height, width]
        """
        # Make a copy to avoid modifying the original
        seq = seq_tensor.clone()
        
        # Apply color jitter with fixed parameters for the entire sequence
        if random.random() < self.color_jitter_prob:
            # Sample fixed parameters once per sequence
            brightness_factor = random.uniform(1.0 - self.brightness_range, 1.0 + self.brightness_range)
            contrast_factor = random.uniform(1.0 - self.contrast_range, 1.0 + self.contrast_range)
            saturation_factor = random.uniform(1.0 - self.saturation_range, 1.0 + self.saturation_range)
            hue_factor = random.uniform(-self.hue_range, self.hue_range)
            
            # Create fixed transforms
            color_transform = T.ColorJitter(
                brightness=(brightness_factor, brightness_factor),
                contrast=(contrast_factor, contrast_factor),
                saturation=(saturation_factor, saturation_factor),
                hue=(hue_factor, hue_factor)
            )
            
            # Apply to each frame
            for i in range(seq.size(0)):
                seq[i] = color_transform(seq[i])
        
        # Apply consistent noise to the entire sequence
        if random.random() < self.noise_prob:
            # Generate noise with the same pattern (but different intensity) for all frames
            noise_base = torch.randn_like(seq[0]) * self.noise_std
            
            # Apply slightly varied noise to maintain some temporal difference
            for i in range(seq.size(0)):
                # Vary noise slightly (90-110% of base noise) to avoid perfect correlation
                # while maintaining strong consistency
                noise_factor = random.uniform(0.9, 1.1)
                seq[i] = seq[i] + noise_base * noise_factor
                
            # Clamp to valid range [0, 1] if working with normalized images
            seq = torch.clamp(seq, 0, 1)
            
        return seq


class GeometricSequenceAugmenter:
    """
    Applies consistent geometric augmentations across a sequence of frames.
    Uses small perturbations that maintain the geometric consistency needed for visual odometry.
    """
    def __init__(self, 
                 rotation_prob=0.5,
                 rotation_max_degrees=2.0,  # Small rotation to preserve geometric consistency
                 perspective_prob=0.3,
                 perspective_scale=0.05,    # Small perspective change
                 motion_blur_prob=0.3,
                 motion_blur_kernel=3):     # Small motion blur
        """
        Args:
            rotation_prob: Probability of applying rotation
            rotation_max_degrees: Maximum rotation in degrees
            perspective_prob: Probability of applying perspective transformation
            perspective_scale: Scale of perspective distortion
            motion_blur_prob: Probability of applying motion blur
            motion_blur_kernel: Size of motion blur kernel
        """
        self.rotation_prob = rotation_prob
        self.rotation_max_degrees = rotation_max_degrees
        self.perspective_prob = perspective_prob
        self.perspective_scale = perspective_scale
        self.motion_blur_prob = motion_blur_prob
        self.motion_blur_kernel = motion_blur_kernel
    
    def apply_rotation(self, seq_tensor):
        """Apply consistent small rotation to preserve geometric consistency."""
        # Get random rotation angle
        angle = random.uniform(-self.rotation_max_degrees, self.rotation_max_degrees)
        
        # Apply to each frame
        result = []
        for i in range(seq_tensor.size(0)):
            # Convert to PIL
            img = T.ToPILImage()(seq_tensor[i])
            # Apply rotation
            rotated = T.functional.rotate(img, angle, fill=0)
            # Convert back to tensor
            result.append(T.ToTensor()(rotated))
        
        return torch.stack(result)
    
    def apply_perspective(self, seq_tensor):
        """Apply consistent small perspective transformation."""
        # Generate perspective parameters (same for all frames)
        height, width = seq_tensor.size(2), seq_tensor.size(3)
        
        # Define 8 points (4 corners before and after transformation)
        # Keep distortion small to maintain geometric consistency
        startpoints = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
        
        # Apply small random distortion to corners
        scale = self.perspective_scale
        dx1, dy1 = random.uniform(-scale, scale) * width, random.uniform(-scale, scale) * height
        dx2, dy2 = random.uniform(-scale, scale) * width, random.uniform(-scale, scale) * height
        dx3, dy3 = random.uniform(-scale, scale) * width, random.uniform(-scale, scale) * height
        dx4, dy4 = random.uniform(-scale, scale) * width, random.uniform(-scale, scale) * height
        
        endpoints = [
            [0 + dx1, 0 + dy1], 
            [width - 1 + dx2, 0 + dy2], 
            [width - 1 + dx3, height - 1 + dy3], 
            [0 + dx4, height - 1 + dy4]
        ]
        
        # Apply consistent perspective to each frame
        result = []
        for i in range(seq_tensor.size(0)):
            # Convert to PIL
            img = T.ToPILImage()(seq_tensor[i])
            # Apply perspective
            transformed = T.functional.perspective(img, startpoints, endpoints, fill=0)
            # Convert back to tensor
            result.append(T.ToTensor()(transformed))
        
        return torch.stack(result)
    
    def apply_motion_blur(self, seq_tensor):
        """Apply subtle motion blur to simulate camera motion."""
        # Generate blur kernel (same direction for all frames to maintain consistency)
        kernel_size = self.motion_blur_kernel
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure odd kernel size
            
        # Generate random angle for motion
        angle = random.uniform(0, 180)
        rad = np.deg2rad(angle)
        
        # Create motion blur kernel
        kernel = np.zeros((kernel_size, kernel_size))
        center = kernel_size // 2
        
        # Draw a line in the kernel
        for i in range(kernel_size):
            offset = i - center
            x = int(center + offset * np.cos(rad))
            y = int(center + offset * np.sin(rad))
            
            if 0 <= x < kernel_size and 0 <= y < kernel_size:
                kernel[y, x] = 1.0
        
        # Normalize kernel
        if kernel.sum() > 0:
            kernel /= kernel.sum()
            
        # Convert kernel to tensor
        kernel_tensor = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        
        # Apply kernel consistently to each frame
        result = []
        
        # Pad and apply convolution
        for i in range(seq_tensor.size(0)):
            frame = seq_tensor[i].unsqueeze(0)  # [1, C, H, W]
            
            # Apply to each channel
            channels = []
            for c in range(frame.size(1)):
                channel = frame[:, c:c+1]  # [1, 1, H, W]
                
                # Apply 2D convolution with motion blur kernel
                blurred = F.conv2d(
                    channel, 
                    kernel_tensor, 
                    padding=kernel_size//2
                )
                
                channels.append(blurred)
            
            # Combine channels and remove batch dimension
            blurred_frame = torch.cat(channels, dim=1).squeeze(0)
            result.append(blurred_frame)
        
        return torch.stack(result)
        
    def __call__(self, seq_tensor):
        """
        Apply geometric augmentations consistently to a sequence.
        
        Args:
            seq_tensor: Sequence of frames tensor [seq_len, channels, height, width]
            
        Returns:
            Augmented sequence tensor [seq_len, channels, height, width]
        """
        # Apply augmentations with their respective probabilities
        seq = seq_tensor.clone()
        
        # Small rotation (preserves geometric consistency)
        if random.random() < self.rotation_prob:
            seq = self.apply_rotation(seq)
        
        # Small perspective change
        if random.random() < self.perspective_prob:
            seq = self.apply_perspective(seq)
        
        # Motion blur
        if random.random() < self.motion_blur_prob:
            seq = self.apply_motion_blur(seq)
            
        return seq


class CutoutSequenceAugmenter:
    """
    Applies consistent cutouts across a sequence of frames.
    Forces the network to rely on global structure rather than specific details.
    """
    def __init__(self, 
                 cutout_prob=0.5,
                 cutout_count=(1, 5),
                 cutout_size_range=(0.05, 0.15)):
        """
        Args:
            cutout_prob: Probability of applying cutouts
            cutout_count: Range of number of cutouts to apply (min, max)
            cutout_size_range: Range of cutout size as fraction of image (min, max)
        """
        self.cutout_prob = cutout_prob
        self.cutout_count = cutout_count
        self.cutout_size_range = cutout_size_range
    
    def __call__(self, seq_tensor):
        """
        Apply consistent cutouts across the sequence.
        
        Args:
            seq_tensor: Sequence of frames tensor [seq_len, channels, height, width]
            
        Returns:
            Augmented sequence tensor [seq_len, channels, height, width]
        """
        if random.random() > self.cutout_prob:
            return seq_tensor
        
        # Make a copy to avoid modifying the original
        seq = seq_tensor.clone()
        
        height, width = seq.size(2), seq.size(3)
        
        # Determine number of cutouts
        num_cutouts = random.randint(self.cutout_count[0], self.cutout_count[1])
        
        # Generate cutout parameters once for the entire sequence
        cutouts = []
        for _ in range(num_cutouts):
            # Determine cutout size
            size_fraction = random.uniform(self.cutout_size_range[0], self.cutout_size_range[1])
            cutout_h = int(height * size_fraction)
            cutout_w = int(width * size_fraction)
            
            # Determine cutout position
            x = random.randint(0, width - cutout_w) if cutout_w < width else 0
            y = random.randint(0, height - cutout_h) if cutout_h < height else 0
            
            cutouts.append((x, y, cutout_w, cutout_h))
        
        # Apply cutouts consistently to each frame
        for i in range(seq.size(0)):
            for x, y, cutout_w, cutout_h in cutouts:
                seq[i, :, y:y+cutout_h, x:x+cutout_w] = 0.0
        
        return seq


class WeatherSequenceAugmenter:
    """
    Applies consistent weather-like effects across a sequence of frames.
    Simulates environmental conditions like fog, rain streaks, and illumination changes.
    """
    def __init__(self, 
                 fog_prob=0.2,
                 fog_density_range=(0.1, 0.3),
                 streak_prob=0.2,
                 streak_count_range=(10, 30),
                 streak_width_range=(1, 2),
                 streak_intensity_range=(0.05, 0.15),
                 vignette_prob=0.2,
                 vignette_range=(0.5, 0.9)):
        """
        Args:
            fog_prob: Probability of applying fog effect
            fog_density_range: Range of fog density (min, max)
            streak_prob: Probability of applying rain/snow streaks
            streak_count_range: Range of streak count (min, max)
            streak_width_range: Range of streak width in pixels (min, max)
            streak_intensity_range: Range of streak intensity (min, max)
            vignette_prob: Probability of applying vignette
            vignette_range: Range of vignette size (min, max)
        """
        self.fog_prob = fog_prob
        self.fog_density_range = fog_density_range
        self.streak_prob = streak_prob
        self.streak_count_range = streak_count_range
        self.streak_width_range = streak_width_range
        self.streak_intensity_range = streak_intensity_range
        self.vignette_prob = vignette_prob
        self.vignette_range = vignette_range
    
    def apply_fog(self, seq_tensor):
        """Apply consistent fog effect to the sequence."""
        # Get random fog density and color
        density = random.uniform(self.fog_density_range[0], self.fog_density_range[1])
        fog_color = torch.tensor([random.uniform(0.7, 1.0), 
                                  random.uniform(0.7, 1.0), 
                                  random.uniform(0.7, 1.0)])
        
        # Apply fog consistently to each frame
        result = []
        for i in range(seq_tensor.size(0)):
            # Blend frame with fog color based on density
            fog_frame = seq_tensor[i] * (1 - density) + fog_color.view(3, 1, 1) * density
            result.append(fog_frame)
        
        return torch.stack(result)
    
    def apply_streaks(self, seq_tensor):
        """Apply consistent rain or snow streaks to the sequence."""
        height, width = seq_tensor.size(2), seq_tensor.size(3)
        
        # Generate streak parameters (same for all frames in sequence)
        num_streaks = random.randint(self.streak_count_range[0], self.streak_count_range[1])
        streak_width = random.randint(self.streak_width_range[0], self.streak_width_range[1])
        streak_intensity = random.uniform(self.streak_intensity_range[0], self.streak_intensity_range[1])
        
        # Generate streak positions and angles (consistent across frames)
        streaks = []
        for _ in range(num_streaks):
            x = random.randint(0, width)
            y1 = random.randint(-height // 2, height // 2)
            length = random.randint(height // 6, height // 3)
            angle = random.uniform(-0.2, 0.2)  # Small angle variation
            streaks.append((x, y1, length, angle, streak_width))
        
        # Apply streaks consistently to each frame with small position shifts for realism
        result = []
        for frame_idx in range(seq_tensor.size(0)):
            frame = seq_tensor[frame_idx].clone()
            
            # Apply each streak
            for idx, (x, y1, length, angle, width) in enumerate(streaks):
                # Move streaks down slightly in each frame for a rain effect
                y_offset = (frame_idx * 10) % height
                
                # Update y position with offset
                y1_offset = (y1 + y_offset) % height
                
                # Create streak mask
                for i in range(length):
                    # Calculate streak position with angle
                    y = y1_offset + i
                    x_shift = int(i * angle)
                    x_pos = (x + x_shift) % width
                    
                    # Ensure y is within bounds
                    if 0 <= y < height:
                        # Add streak with width
                        for w in range(-width // 2, width // 2 + 1):
                            x_w = (x_pos + w) % width
                            # Add streak with slight intensity variation
                            intensity_var = streak_intensity * random.uniform(0.8, 1.2)
                            frame[:, y, x_w] += intensity_var
            
            # Clamp values to valid range
            frame = torch.clamp(frame, 0, 1)
            result.append(frame)
        
        return torch.stack(result)
    
    def apply_vignette(self, seq_tensor):
        """Apply consistent vignette effect to the sequence."""
        height, width = seq_tensor.size(2), seq_tensor.size(3)
        
        # Generate vignette parameters
        vignette_size = random.uniform(self.vignette_range[0], self.vignette_range[1])
        
        # Create vignette mask
        center_x, center_y = width // 2, height // 2
        x = torch.arange(width).view(1, -1).expand(height, -1)
        y = torch.arange(height).view(-1, 1).expand(-1, width)
        
        # Calculate distance from center (normalized)
        dist = torch.sqrt(((x - center_x) / width * 2) ** 2 + 
                         ((y - center_y) / height * 2) ** 2)
        
        # Create vignette mask
        vignette_mask = torch.clamp(1 - dist / vignette_size, 0, 1)
        
        # Apply vignette consistently to each frame
        result = []
        for i in range(seq_tensor.size(0)):
            # Apply vignette mask to each channel
            vignette_frame = seq_tensor[i] * vignette_mask.unsqueeze(0)
            result.append(vignette_frame)
        
        return torch.stack(result)
    
    def __call__(self, seq_tensor):
        """
        Apply weather augmentations consistently to a sequence.
        
        Args:
            seq_tensor: Sequence of frames tensor [seq_len, channels, height, width]
            
        Returns:
            Augmented sequence tensor [seq_len, channels, height, width]
        """
        seq = seq_tensor.clone()
        
        # Apply fog
        if random.random() < self.fog_prob:
            seq = self.apply_fog(seq)
        
        # Apply rain/snow streaks
        if random.random() < self.streak_prob:
            seq = self.apply_streaks(seq)
        
        # Apply vignette
        if random.random() < self.vignette_prob:
            seq = self.apply_vignette(seq)
        
        return seq


class SequentialAugmentationPipeline:
    """
    Combines multiple sequence augmenters into a single pipeline.
    Ensures all augmentations are applied consistently across the sequence.
    """
    def __init__(self, augmenters=None):
        """
        Args:
            augmenters: List of augmenter objects to apply
        """
        self.augmenters = augmenters or []
    
    def add_augmenter(self, augmenter):
        """Add an augmenter to the pipeline."""
        self.augmenters.append(augmenter)
    
    def __call__(self, seq_tensor):
        """
        Apply all augmentations in the pipeline consistently to a sequence.
        
        Args:
            seq_tensor: Sequence of frames tensor [seq_len, channels, height, width]
            
        Returns:
            Augmented sequence tensor [seq_len, channels, height, width]
        """
        seq = seq_tensor.clone()
        
        # Apply each augmenter in sequence
        for augmenter in self.augmenters:
            seq = augmenter(seq)
        
        return seq


def create_vo_augmentation_pipeline(
    # Color augmentation params
    color_jitter_prob=0.8,
    brightness_range=0.3,
    contrast_range=0.3,
    saturation_range=0.3,
    hue_range=0.1,
    noise_prob=0.4,
    noise_std=0.03,
    
    # Geometric augmentation params
    rotation_prob=0.5,
    rotation_max_degrees=2.0,
    perspective_prob=0.3,
    perspective_scale=0.05,
    motion_blur_prob=0.3,
    motion_blur_kernel=3,
    
    # Cutout augmentation params
    cutout_prob=0.4,
    cutout_count=(1, 3),
    cutout_size_range=(0.05, 0.15),
    
    # Weather augmentation params
    fog_prob=0.1,
    fog_density_range=(0.1, 0.2),
    streak_prob=0.1,
    streak_count_range=(10, 20),
    vignette_prob=0.2
):
    """
    Create a comprehensive augmentation pipeline for visual odometry.
    All augmentations maintain geometric and temporal consistency across the sequence.
    
    Returns:
        SequentialAugmentationPipeline: Complete augmentation pipeline
    """
    # Create pipeline
    pipeline = SequentialAugmentationPipeline()
    
    # Add color augmenter
    color_augmenter = SequenceAugmenter(
        color_jitter_prob=color_jitter_prob,
        brightness_range=brightness_range,
        contrast_range=contrast_range,
        saturation_range=saturation_range,
        hue_range=hue_range,
        noise_prob=noise_prob,
        noise_std=noise_std
    )
    pipeline.add_augmenter(color_augmenter)
    
    # Add geometric augmenter
    geometric_augmenter = GeometricSequenceAugmenter(
        rotation_prob=rotation_prob,
        rotation_max_degrees=rotation_max_degrees,
        perspective_prob=perspective_prob,
        perspective_scale=perspective_scale,
        motion_blur_prob=motion_blur_prob,
        motion_blur_kernel=motion_blur_kernel
    )
    pipeline.add_augmenter(geometric_augmenter)
    
    # Add cutout augmenter
    cutout_augmenter = CutoutSequenceAugmenter(
        cutout_prob=cutout_prob,
        cutout_count=cutout_count,
        cutout_size_range=cutout_size_range
    )
    pipeline.add_augmenter(cutout_augmenter)
    
    # Add weather augmenter
    weather_augmenter = WeatherSequenceAugmenter(
        fog_prob=fog_prob,
        fog_density_range=fog_density_range,
        streak_prob=streak_prob,
        streak_count_range=streak_count_range,
        vignette_prob=vignette_prob
    )
    pipeline.add_augmenter(weather_augmenter)
    
    return pipeline