# Inspect FlowNet weights structure
import torch
from params import par

flownet_path = par.pretrained_flownet
checkpoint = torch.load(flownet_path, map_location='cpu')

# Check if weights are stored under 'state_dict'
if 'state_dict' in checkpoint:
    pretrained_weights = checkpoint['state_dict']
else:
    pretrained_weights = checkpoint

# Print the first 20 keys to see the layer naming pattern
print("FlowNet weights keys:")
for i, key in enumerate(list(pretrained_weights.keys())[:20]):
    print(f"  - {key}")