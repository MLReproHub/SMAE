import torch
from matplotlib import pyplot as plt


def visualize_reconstruction_batch(model, x_b, vis_transforms=None, n_rows=2, show=False, device='cpu', plot_mask=False,
                                   blocks=None, return_img=False):
    fig, axes = plt.subplots(nrows=n_rows, ncols=3 if plot_mask else 2, figsize=(9, 4 * n_rows))
    imgs = []
    # If there is only one row
    if len(axes.shape) == 1:
        axes = [axes]
    model.eval()
    with torch.no_grad():
        for i, ax in enumerate(axes):
            x = x_b[i]
            x_hat, _, x_patches, mask_indices = model(x.unsqueeze(0).to(device), blocks)
            x_hat = x_hat.detach().squeeze()
            if plot_mask:
                # Fill masked patches with gray
                x_patches.scatter_(dim=1,
                                   index=mask_indices[:, :, :x_patches.shape[-1]],
                                   src=0.5 * torch.ones(size=x_patches.shape, device=x_patches.device))
                # Fold into original image dimensions
                x_masked = model.fold(x_patches.swapaxes(1, 2), x.shape[1:]).squeeze()
            if vis_transforms is not None:
                x = torch.clip(vis_transforms(x), 0, 1)
                x_hat = torch.clip(vis_transforms(x_hat), 0, 1)
                if plot_mask:
                    x_masked = torch.clip(vis_transforms(x_masked), 0, 1)

            titles = ['Reconstruction', 'Ground Truth']
            images = [x_hat, x]
            if plot_mask:
                titles.insert(0, 'Masked Input')
                images.insert(0, x_masked)
            for column, title, img in zip(ax, titles, images):
                img = img.cpu().permute(1, 2, 0)
                column.imshow(img)
                column.axis('off')
                column.set_title(title)
                imgs.append(img)

    fig.tight_layout()
    if show:
        plt.show()
    if return_img:
        return imgs
    return fig


def visualize_reconstruction(model, dl_train, vis_transforms=None, n_rows=2, show=False, device='cpu', plot_mask=False,
                             blocks=None):
    x_b, _ = next(iter(dl_train))
    return visualize_reconstruction_batch(model=model, x_b=x_b, vis_transforms=vis_transforms, n_rows=n_rows, show=show,
                                          device=device, plot_mask=plot_mask, blocks=blocks)


def plot_single_image(dl_train, vis_transforms=None, show=False):
    fig, ax = plt.subplots(figsize=(6, 6))

    x_b, _ = next(iter(dl_train))
    x = x_b[0]

    if vis_transforms is not None:
        x = torch.clip(vis_transforms(x), 0, 1)

    ax.imshow(x.cpu().permute(1, 2, 0))
    ax.axis('off')
    ax.set_title("Training Image")

    fig.tight_layout()
    if show:
        plt.show()
    return fig


def show_example(ds, index):
    print(f"Label: {ds[index]['label']}")
    plt.figure()
    plt.imshow(ds[index]['image'])
    plt.show()
