import sys
from contextlib import nullcontext
from enum import Enum

import torch
from matplotlib import pyplot as plt
from torch import nn
from tqdm.auto import tqdm

from loss.combined import CombinedLoss
from loss.perceptual import Perceptual as PerceptualLoss
from loss.uncertainty import UncertaintyWeightedLoss
from model.layer import ReorderToBlockWiseMask
from utilities.misc import prepare_blocks
from utilities.neptune import NeptuneRun
from utilities.train import TrainingSetup
from utilities.visualize import visualize_reconstruction, plot_single_image

Partition = Enum('Partition', ('TRAIN', 'VALID'))


class StatisticsTracker:
    def __init__(self, neptune_run: NeptuneRun):
        self.neptune_run = neptune_run

    def start_next_epoch(self, partition: Partition, num_batches: int):
        self.epoch_partition_name = partition.name.lower()
        self.epoch_losses = torch.zeros(num_batches)

    def log_epoch_loss(self, example_index, loss):
        self.epoch_losses[example_index] = loss
        self.neptune_run.log(f"{self.epoch_partition_name}/batch/loss", loss)

    def finish_epoch(self, accuracy: None or float = None):
        loss_name = f"{self.epoch_partition_name}/loss"
        loss = self.epoch_losses.mean()
        setattr(self, "last_" + loss_name, loss)
        self.neptune_run.log(loss_name, loss)

        if accuracy is not None:
            accuracy_name = f"{self.epoch_partition_name}/accuracy"
            setattr(self, "last_" + accuracy_name, accuracy)
            self.neptune_run.log(accuracy_name, accuracy)

    def log_learning_rate(self, learning_rate):
        self.last_learning_rate = learning_rate
        self.neptune_run.log("schedule/learning_rate", learning_rate)

    def get_last_valid_loss(self):
        return getattr(self, "last_valid/loss")

    def log_reconstruction(self, partition, figure):
        self.neptune_run.log(f"{partition.name.lower()}/reconstruction", figure)

    # Returns a string describing the latest metrics of the run
    def metrics(self):
        desc = f"lr: {self.last_learning_rate:.3E}|"
        for partition in Partition:
            p_name = partition.name.lower()

            try:
                last_loss = getattr(self, f"last_{p_name}/loss")
            # If there are no stats for validation set, only print test stats.
            except AttributeError:
                continue
            desc += f"{p_name}_loss: {last_loss:.3f}|"

            last_accuracy_name = f"last_{p_name}/accuracy"
            if hasattr(self, last_accuracy_name):
                last_accuracy = getattr(self, last_accuracy_name)
                desc += f"{p_name}_accuracy: {last_accuracy:.3f}|"

        return desc


@NeptuneRun.handle_connection_error
def pre_train(model, training_setup: TrainingSetup, train_dataloader, val_dataloader, neptune_run: NeptuneRun,
              capture=None):
    optim: torch.optim.Adam = training_setup.optimizer
    loss_fn = training_setup.loss_function
    if isinstance(loss_fn, UncertaintyWeightedLoss):
        optim.add_param_group({
            'params': [loss_fn.log_var1, loss_fn.log_var2],
            'lr': optim.param_groups[0]['lr']
        })

    blocks = None
    if isinstance(model.reorder_tokens, ReorderToBlockWiseMask):
        # Prepare mask blocks
        blocks = prepare_blocks(model, next(iter(train_dataloader))[0][0].unsqueeze(0), training_setup.device, n=10000)

    tracker = StatisticsTracker(neptune_run)

    # Add one initial epoch for computing the metrics before training starts.
    for t in range(training_setup.num_epochs + 1):
        desc = f"Epoch: {t:3d}|{tracker.metrics() if t != 0 else 'Computing initial metrics'}"
        train_pbar = tqdm(train_dataloader, desc=desc)
        model.train()
        tracker.start_next_epoch(Partition.TRAIN, len(train_dataloader))
        # Training loop
        for i, (x, _) in enumerate(train_pbar):
            if type(x) == list:
                x = x[0]
            x = x.to(training_setup.device)
            with torch.no_grad() if t == 0 else nullcontext():
                reconstruction, reconstructed_patches, original_patches, mask_indices = model(x, blocks)
                if isinstance(loss_fn, PerceptualLoss):
                    loss = loss_fn(reconstruction, x)
                elif isinstance(loss_fn, CombinedLoss) or isinstance(loss_fn, UncertaintyWeightedLoss):
                    loss, pixel_loss, perceptual_loss = loss_fn(reconstruction, x, reconstructed_patches,
                                                                original_patches)
                    neptune_run.log(f"train/batch/pixel_loss", pixel_loss)
                    neptune_run.log(f"train/batch/perceptual_loss", perceptual_loss)
                    if isinstance(loss_fn, UncertaintyWeightedLoss):
                        neptune_run.log(f"train/batch/sigma_pixel", loss_fn.log_var1.detach())
                        neptune_run.log(f"train/batch/sigma_perceptual", loss_fn.log_var2.detach())
                    neptune_run.log(f"train/batch/w_pixel", loss_fn.w_pixel)
                    neptune_run.log(f"train/batch/w_perceptual", loss_fn.w_perceptual)
                else:
                    loss = loss_fn(reconstructed_patches, original_patches, mask_indices)
                tracker.log_epoch_loss(i, loss.item())
            if t > 0:
                optim.zero_grad()
                loss.backward()
                optim.step()

        tracker.finish_epoch()

        val_pbar = tqdm(val_dataloader, desc="Evaluating on validation set", leave=False)
        model.eval()
        tracker.start_next_epoch(Partition.VALID, len(val_dataloader))
        # Validation loop
        for i, (x, _) in enumerate(val_pbar):
            if type(x) == list:
                x = x[0]
            x = x.to(training_setup.device)
            with torch.no_grad():
                reconstruction, reconstructed_patches, original_patches, mask_indices = model(x, blocks)
                if isinstance(loss_fn, PerceptualLoss):
                    loss = loss_fn(reconstruction, x)
                elif isinstance(loss_fn, CombinedLoss) or isinstance(loss_fn, UncertaintyWeightedLoss):
                    loss, pixel_loss, perceptual_loss = loss_fn(reconstruction, x, reconstructed_patches,
                                                                original_patches)
                else:
                    loss = loss_fn(reconstructed_patches, original_patches, mask_indices)
                tracker.log_epoch_loss(i, loss.item())

        tracker.finish_epoch()
        tracker.log_learning_rate(optim.param_groups[0]['lr'])

        # TODO: Refactor to take checkpoints during training at proper place
        if t % 50 == 0 and capture is not None:
            capture(model, training_setup, t)

        for partition, dataloader in zip(Partition, (train_dataloader, val_dataloader)):
            fig = visualize_reconstruction(model, dataloader, dataloader.vis_transforms,
                                           show=neptune_run.is_debug_mode(),
                                           device=training_setup.device,
                                           n_rows=2,
                                           plot_mask=True,
                                           blocks=blocks)

            tracker.log_reconstruction(partition, fig)

            # To avoid memory leak
            plt.close(fig)

        if t > 0:
            training_setup.lr_scheduler.step()

    print(f"Final epoch:{'|':>5}{tracker.metrics()}", file=sys.stderr)


@NeptuneRun.handle_connection_error
def fine_tune(model: nn.Module, training_setup: TrainingSetup, train_dataloader, val_dataloader,
              neptune_run: NeptuneRun, capture=None):
    optim = training_setup.optimizer
    loss_fn = training_setup.loss_function

    tracker = StatisticsTracker(neptune_run)

    # Add one initial epoch for computing the metrics before training starts.
    for t in range(training_setup.num_epochs + 1):
        desc = f"Epoch: {t:3d}|{tracker.metrics() if t != 0 else 'Computing initial metrics'}"
        train_pbar = tqdm(train_dataloader, desc=desc)
        correct = 0
        # Training loop
        tracker.start_next_epoch(Partition.TRAIN, len(train_dataloader))
        for i, (x, y) in enumerate(train_pbar):
            if type(x) == list:
                x = x[0]
            x = x.to(training_setup.device)
            y = y.to(training_setup.device)
            with torch.no_grad() if t == 0 else nullcontext():
                pred = model(x)
                loss = loss_fn(pred, y)
                tracker.log_epoch_loss(i, loss.item())
                # noinspection PyUnresolvedReferences
                correct += (pred.argmax(dim=-1) == y.argmax(dim=-1)).sum()
            if t > 0:
                optim.zero_grad()
                loss.backward()
                optim.step()

        accuracy = correct / len(train_dataloader.dataset)
        tracker.finish_epoch(accuracy)
        tracker.log_learning_rate(optim.param_groups[0]['lr'])

        val_pbar = tqdm(val_dataloader, desc="Evaluating on validation set", leave=False)
        correct = 0
        tracker.start_next_epoch(Partition.VALID, len(val_dataloader))
        # Validation loop
        for i, (x, y) in enumerate(val_pbar):
            if type(x) == list:
                x = x[0]
            x = x.to(training_setup.device)
            y = y.to(training_setup.device)
            with torch.no_grad():
                pred = model(x)
                loss = loss_fn(pred, y)
                tracker.log_epoch_loss(i, loss.item())
                # noinspection PyUnresolvedReferences
                correct += (pred.argmax(dim=-1) == y.argmax(dim=-1)).sum()

        accuracy = correct / len(val_dataloader.dataset)
        tracker.finish_epoch(accuracy)

        if t > 0 and training_setup.scheduler is not None:
            try:
                training_setup.lr_scheduler.step(tracker.get_last_valid_loss())
            except TypeError:
                training_setup.lr_scheduler.step()

        # TODO: Refactor to take checkpoints during training at proper place
        if t % 50 == 0 and capture is not None:
            capture(model, training_setup, t)

        # Log training image
        fig = plot_single_image(train_dataloader, vis_transforms=train_dataloader.vis_transforms,
                                show=neptune_run.is_debug_mode())
        tracker.log_reconstruction(Partition.TRAIN, fig)
        # To avoid memory leak
        plt.close(fig)

    print(f"Final epoch:{'|':>5}{tracker.metrics()}", file=sys.stderr)
