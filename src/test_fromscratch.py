import os
import torch

# Verifies two checkpoints, taken before and after training from scratch,
# and determines which parameters values are unchanged after training

before_checkpoint_filename = "mae_e7d2_128_fromscratch_0.pth"
after_checkpoint_filename = "mae_e7d2_128_fromscratch.pth"

os.chdir("../checkpoints")

before_training = torch.load(before_checkpoint_filename)['model']
after_training = torch.load(after_checkpoint_filename)['model']

all_are_equal = True

for key in before_training.keys():
    before_value = before_training[key]
    after_value = after_training[key]
    is_equal = torch.allclose(before_value, after_value)
    if is_equal:
        all_are_equal = False
        print(f"All values under key {key} are equal before and after training")

if all_are_equal:
    print("No keys have only unchanged values")
