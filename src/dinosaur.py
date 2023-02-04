"""
This is an adaptation of the official DINO code.
https://github.com/facebookresearch/dino
"""

import dataclasses
import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.transforms import Normalize
from tqdm import tqdm

import dataset.tiny_imagenet
from model.mae import ClassifierEncoder
from train import Partition, StatisticsTracker
from utilities import dino as utils
from utilities.config import capture_checkpoint
from utilities.neptune import NeptuneRun


def run(model1, model2, device, cr, training_setup):
    args = DinoConfig()
    # Make sure the models are different
    assert get_param(model1) != get_param(model2), "models are not different"
    student_vit = ClassifierEncoder(model1, freeze_n_layers=None)
    teacher_vit = ClassifierEncoder(model2, freeze_n_layers=None)
    student = create_dino_net(student_vit,
                              out_dim=args.out_dim,
                              hidden_dim=args.hidden_dim,
                              bottleneck_dim=args.bottleneck_dim,
                              norm_last_layer=args.norm_last_layer).to(device)
    teacher = create_dino_net(teacher_vit,
                              out_dim=args.out_dim,
                              hidden_dim=args.hidden_dim,
                              bottleneck_dim=args.bottleneck_dim,
                              norm_last_layer=args.norm_last_layer,
                              requires_grad=False).to(device)

    data_loader = dataset.tiny_imagenet.TinyImagenetDataLoader(train=True,
                                                               batch_size=args.batch_size_per_gpu,
                                                               device=device,
                                                               extra_transforms='dino',
                                                               global_crops_scale=args.global_crops_scale,
                                                               local_crops_scale=args.local_crops_scale,
                                                               local_crops_number=args.local_crops_number,
                                                               num_workers=1,
                                                               pin_memory=True)

    dino_loss = DINOLoss(
        args.out_dim,
        args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    ).to(device)

    with NeptuneRun() as neptune_run:
        pre_train_dino(student, teacher, args, data_loader, dino_loss, neptune_run)
        model = student.backbone.model
        # Will save as a pretrain MAE checkpoint
        capture_checkpoint(model, training_setup, cr, "", neptune_run)


def create_dino_net(encoder, out_dim, hidden_dim, bottleneck_dim, use_bn=False, norm_last_layer=False,
                    requires_grad=True):
    net = utils.MultiCropWrapper(encoder, utils.DINOHead(
        in_dim=encoder.d_model,
        out_dim=out_dim,
        use_bn=use_bn,
        # TODO Try enabling this if training is unstable
        norm_last_layer=norm_last_layer,
        hidden_dim=hidden_dim,
        bottleneck_dim=bottleneck_dim,
    ))
    if not requires_grad:
        for p in net.parameters():
            p.requires_grad = False
    return net


def get_param(model):
    return model.decoder.enc_layers[0].feed_forward[0].weight[0, 0]


@NeptuneRun.handle_connection_error
def pre_train_dino(student, teacher, args, train_dataloader, dino_loss, neptune_run: NeptuneRun,
                   capture=None):
    tracker = StatisticsTracker(neptune_run)

    # momentum parameter is increased to 1 during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(base_value=args.momentum_teacher, final_value=1,
                                               epochs=args.epochs, niter_per_ep=len(train_dataloader))

    params_groups = utils.get_params_groups(student)
    optimizer = torch.optim.AdamW(params_groups)
    lr_schedule = utils.cosine_scheduler(
        base_value=args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        final_value=args.min_lr,
        epochs=args.epochs,
        niter_per_ep=len(train_dataloader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        base_value=args.weight_decay,
        final_value=args.weight_decay_end,
        epochs=args.epochs,
        niter_per_ep=len(train_dataloader),
    )

    for epoch in range(args.epochs):
        train_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}| {tracker.metrics() if epoch != 0 else ''}")

        tracker.start_next_epoch(Partition.TRAIN, len(train_dataloader))
        for it, (images, _) in enumerate(train_pbar):
            # update weight decay and learning rate according to their schedule
            total_it = len(train_dataloader) * epoch + it  # global training iteration
            for i, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = lr_schedule[total_it]
                if i == 0:  # only the first group is regularized
                    param_group["weight_decay"] = wd_schedule[total_it]

            # move images to gpu
            images = [im.cuda() for im in images]
            # teacher and student forward passes + compute dino loss
            teacher_output = teacher(images[:2])  # only the 2 global views pass through the teacher
            student_output = student(images)
            loss = dino_loss(student_output, teacher_output, epoch)

            if not math.isfinite(loss.item()):
                print("Loss is {}, stopping training".format(loss.item()))
                return

            # student update
            optimizer.zero_grad()
            param_norms = None
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()

            # EMA update for the teacher
            with torch.no_grad():
                m = momentum_schedule[total_it]  # momentum parameter
                for param_q, param_k in zip(student.parameters(), teacher.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

            tracker.log_epoch_loss(it, loss.item())

        tracker.finish_epoch()
        tracker.log_learning_rate(optimizer.param_groups[0]['lr'])


@dataclasses.dataclass
class DinoConfig:
    """
    Holds the default values if not specified otherwise
    """
    momentum_teacher = 0.9995
    # They account for linear scaling rule in the training loop
    lr = 0.0001
    # Custom value according to our batch size
    batch_size_per_gpu = 128
    min_lr = 1e-6
    epochs = 100
    warmup_epochs = 10
    weight_decay = 0.04
    weight_decay_end = 0.4

    out_dim = 1024
    hidden_dim = 512
    bottleneck_dim = 128
    # Disable multi-crop
    local_crops_number = 0
    warmup_teacher_temp = 0.02
    teacher_temp = 0.04
    warmup_teacher_temp_epochs = 30
    norm_last_layer = True

    clip_grad = 3.0
    freeze_last_layer = 1

    # Adapt to disabled multi-crop
    global_crops_scale = (0.6, 1.)
    local_crops_scale = (0.05, 0.4)


# Customized for Tiny Imagenet
class DataAugmentationDINO:
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number, size=64):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            # Tiny IN values
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(size - 20, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops


# Non-distributed version of DINOLoss
class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        # dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output))

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
