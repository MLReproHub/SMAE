from matplotlib import pyplot as plt
import matplotlib.font_manager

from dataset.tiny_imagenet import TinyImagenetDataset
from utilities.config import ConfigReader
from utilities.visualize import visualize_reconstruction_batch

MODEL_CONFIG = 'e7d2_128'
DS_INDEX = 1034

model_random = ConfigReader.load_model_from_checkpoint('mae', MODEL_CONFIG)
model_block = ConfigReader.load_model_from_checkpoint('mae-blockmask', MODEL_CONFIG)

ds = TinyImagenetDataset(train=False)
x_b = ds[DS_INDEX][0].unsqueeze(0)

imgs = visualize_reconstruction_batch(model_random, x_b, ds.vis_transforms(), n_rows=1, show=False,
                                      plot_mask=True, return_img=True)
random_masked = imgs[0]
random_recon = imgs[1]

imgsb = visualize_reconstruction_batch(model_block, x_b, ds.vis_transforms(), n_rows=1, show=False,
                                       plot_mask=True, return_img=True)
block_masked, block_recon, gt = imgsb

for font in matplotlib.font_manager.findSystemFonts('/usr/share/fonts/ttf/Source_Serif_Pro'):
    matplotlib.font_manager.fontManager.addfont(font)
plt.rcParams['font.family'] = 'Source Serif Pro'

fig = plt.figure(figsize=(10, 2))
ax_random = fig.subplots(1, 2, gridspec_kw=dict(left=0, right=0.33, wspace=0.2))
ax_masked = fig.subplots(1, 2, gridspec_kw=dict(left=0.42, right=0.75, wspace=0.2))
ax_gt = fig.subplots(1, 1, gridspec_kw=dict(left=0.85, right=0.99, wspace=0.4))

ax_random[0].imshow(random_masked)
ax_random[0].axis('off')
ax_random[0].set_title('Masked Input', fontsize=9)
ax_random[1].imshow(random_recon)
ax_random[1].axis('off')
ax_random[1].set_title('Reconstruction', fontsize=9)
line1 = plt.Line2D((.375, .375), (.1, .9), color="k", linewidth=1)
fig.add_artist(line1)

ax_masked[0].imshow(block_masked)
ax_masked[0].axis('off')
ax_masked[0].set_title('Blocked Masked Input', fontsize=9)
ax_masked[1].imshow(block_recon)
ax_masked[1].axis('off')
ax_masked[1].set_title('Blocked Reconstruction', fontsize=9)
line2 = plt.Line2D((.80, .80), (.1, .9), color="k", linewidth=1)
fig.add_artist(line2)

ax_gt.imshow(gt)
ax_gt.axis('off')
ax_gt.set_title('Ground Truth', fontsize=9)

plt.tight_layout()
plt.savefig('reconstructions.pdf')
