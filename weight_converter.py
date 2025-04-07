import pickle
import torch
from collections import OrderedDict

d2_path = '/Users/tsuipo/Downloads/vitdet_swin-l_IN21k-sup_coco_cascade-mask.pkl'
# d2_path = 'path/to/your_d2_model_weight.pkl'

with open(d2_path, 'rb') as f:
    d2_weight = pickle.load(f)

print(d2_weight.keys())

new_weight = dict()
new_weight['model'] = OrderedDict()
for k, v in d2_weight['model'].items():
    if 'bottom_up' in k:
        # Convert numpy arrays to torch tensors
        if hasattr(v, 'shape'):  # Check if it's a numpy array
            new_weight['model'][k[19:]] = torch.from_numpy(v)
        else:
            new_weight['model'][k[19:]] = v

torch.save(new_weight, '/Users/tsuipo/Downloads/vitdet_swin-l_IN21k-sup_coco_cascade-mask.pth')
print("Conversion completed. All numpy arrays converted to PyTorch tensors.")