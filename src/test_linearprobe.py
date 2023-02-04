import os
import torch

# Verifies two checkpoints, taken before and after linear probing, such that
# all parameter values in encoder are unchanged after linear probing

before_checkpoint_filename = "mae_e7d2_128.pth"
after_checkpoint_filename = "mae_e7d2_128_linearprobing.pth"

os.chdir("../checkpoints")

before_linearprobe = torch.load(before_checkpoint_filename)['model']
after_linearprobe = torch.load(after_checkpoint_filename)['model']

all_are_equal = True

for key in before_linearprobe.keys():
    if key.startswith("encoder."):
        before_value = before_linearprobe[key]
        after_value = after_linearprobe["0.model." + key]
        is_equal = torch.allclose(before_value, after_value)
        if not is_equal:
            all_are_equal = False
            print(f"Key {key} is not equal before and after training")

if all_are_equal:
    print("All values are equal")
