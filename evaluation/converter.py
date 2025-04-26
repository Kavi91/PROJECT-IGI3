import numpy as np
from scipy.spatial.transform import Rotation as R

def convert_file(input_file, output_file):
    print(f"Converting {input_file} to {output_file}")
    
    # Load the 6-column data
    data = np.loadtxt(input_file)
    print(f"Loaded data with shape {data.shape}")
    
    # Create a new array with 7 columns
    new_data = np.zeros((data.shape[0], 7))
    
    # Copy the position components (first 3 columns)
    new_data[:, 0:3] = data[:, 0:3]
    
    # Process each row
    for i in range(data.shape[0]):
        # Get the rotation components (last 3 columns)
        rot = data[i, 3:6]
        
        # Convert to quaternion (assuming Euler angles in XYZ order)
        try:
            r = R.from_euler('xyz', rot)
            quat = r.as_quat()  # [x, y, z, w]
            new_data[i, 3:7] = quat
        except:
            # If conversion fails, use identity quaternion
            new_data[i, 3:7] = [0.0, 0.0, 0.0, 1.0]
    
    # Save the new 7-column data
    np.savetxt(output_file, new_data, delimiter=' ', fmt='%.18e')
    print(f"Saved 7-column data to {output_file}")
    
    # Verify
    verify = np.loadtxt(output_file)
    print(f"Verified shape: {verify.shape}")

# Convert both files
convert_file('1pose_gt.txt', 'pose_gt.txt')
convert_file('1pose_est.txt', 'pose_est.txt')

print("\nConversion complete. Now run:")
print("python tartanair_evaluator.py pose_gt.txt pose_est.txt")