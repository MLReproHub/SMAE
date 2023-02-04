from typing import List, Optional

import torch
from datasets import load_dataset
from torch.nn.functional import one_hot
from torch.utils.data import Dataset, DataLoader, BatchSampler, RandomSampler
from torchvision.transforms import Compose, Normalize, ToTensor, RandomResizedCrop, InterpolationMode, \
    RandomHorizontalFlip, RandAugment

import dinosaur
from model import layer
from utilities.path import tiny_imagenet_path


def load_tiny_imagenet(train=True, device="cpu"):
    ds = load_dataset(TinyImagenetDataset.HUGGINGFACE_HUB_PATH, split="train" if train else "valid",
                      cache_dir=tiny_imagenet_path)
    return ds.with_format("torch", device=device)


class ToRGBImage:
    def __call__(self, pic):
        return pic.convert('RGB')


class TinyImagenetDataset(Dataset):
    """
    TinyImagenetDataset Class:
    This class is used to access Tiny-ImageNET-200 via HuggingFace API.
    """

    HUGGINGFACE_HUB_PATH = "Maysee/tiny-imagenet"

    def __init__(self, train: bool = True, use_one_hot: bool = False, transforms=None):
        ds = load_dataset(
            self.__class__.HUGGINGFACE_HUB_PATH,
            split='train' if train else 'valid',
            cache_dir=tiny_imagenet_path,
            ignore_verifications=True
        )
        if transforms is None:
            self.transforms = Compose([
                ToRGBImage(),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transforms = transforms
        self.ds = ds
        self.use_one_hot = use_one_hot

    def __getitem__(self, index: int or List[int]):
        image, label = self.ds[index].values()
        if type(index) == list:
            image = [self.transforms(i) for i in image]
            label = [torch.tensor(lbl) if not self.use_one_hot else one_hot(torch.tensor(lbl), 200)
                     for lbl in label]
        else:
            image = self.transforms(image)
            if self.use_one_hot:
                label = one_hot(torch.tensor(label), self.n_classes).float()
        return image, label

    def __len__(self):
        return len(self.ds)

    @property
    def n_classes(self) -> int:
        return 200

    @staticmethod
    def vis_transforms() -> Compose:
        return Compose([
            Normalize(mean=[0., 0., 0.], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
            Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1])
        ])


class TinyImagenetDataLoader(DataLoader):
    """
    TinyImagenetDataLoader Class:
    This class is used to access HuggingFace/Tiny-ImageNET-200 via PyTorch's Dataloading API.
    """

    def __init__(self, train=True, use_one_hot=True, ds_transforms=None, extra_transforms: Optional[str] = None,
                 device: str = 'cpu', use_val: bool = False, val_size=None, **kwargs):
        self.use_one_hot = use_one_hot

        train_ds, val_ds = TinyImagenetDataset(train=train, use_one_hot=use_one_hot, transforms=ds_transforms), None
        if extra_transforms is not None and not use_val:
            if extra_transforms == 'random_crop':
                train_ds.transforms.transforms = [
                                                     RandomHorizontalFlip(p=0.5),
                                                     RandomResizedCrop(size=64, scale=(0.6, 1.0),
                                                                       interpolation=InterpolationMode.BICUBIC),
                                                 ] + train_ds.transforms.transforms
            elif extra_transforms in ['rand_augment', 'auto_augment']:
                # keeping "auto_augment" here for compatibility with older code versions; we only use RandAugment
                ts = train_ds.transforms.transforms.pop(0)
                train_ds.transforms.transforms = [ts] + [
                    RandomHorizontalFlip(p=0.5),
                    RandomResizedCrop(size=64, scale=(0.6, 1.0), ratio=(0.9, 1.1),
                                      interpolation=InterpolationMode.BICUBIC),
                    RandAugment(num_ops=2, magnitude=9),
                ] + train_ds.transforms.transforms
            elif extra_transforms == 'dino':
                # Insert after ToRGB transform
                train_ds.transforms = Compose([
                    ToRGBImage(),
                    dinosaur.DataAugmentationDINO(
                        kwargs.pop('global_crops_scale'),
                        kwargs.pop('local_crops_scale'),
                        kwargs.pop('local_crops_number'),
                    )
                ])

        if train and use_val and val_size is not None:
            ts = len(train_ds)
            vs = int(ts * val_size) if type(val_size) == float and val_size < 1.0 else val_size
            ts -= vs
            train_ds, val_ds = torch.utils.data.random_split(train_ds, [ts, vs])
        ds = train_ds if not use_val or val_ds is None else val_ds
        if 'pin_memory' not in kwargs:
            kwargs['pin_memory'] = device != 'cpu'

        # Dino augmentations expect single images, so doesn't work with batch sampler...
        if extra_transforms != 'dino':
            sampler = BatchSampler(RandomSampler(ds), batch_size=kwargs.pop('batch_size'), drop_last=False)
            super(TinyImagenetDataLoader, self).__init__(dataset=ds, sampler=sampler, collate_fn=self.collate, **kwargs)
        else:
            super(TinyImagenetDataLoader, self).__init__(dataset=ds, **kwargs)

    def collate(self, batch):
        x, y = batch[0]
        if self.use_one_hot:
            return torch.stack(x), torch.stack(y).float()
        return torch.stack(x), torch.stack(y)

    @property
    def vis_transforms(self) -> Compose:
        return TinyImagenetDataset.vis_transforms()

    @property
    def n_classes(self) -> int:
        # noinspection PyUnresolvedReferences
        return 200


class TinyImagenetFastDataLoader(DataLoader):
    DATASETS = {
        'train': None,
        'test': None,
    }

    def __init__(self, train: bool, batch_size: int, device: str, drop_last=False, use_one_hot: bool = True,
                 use_val: bool = False, **kwargs):
        if self.__class__.DATASETS['train' if train else 'test'] is None:
            self.__class__.DATASETS['train' if train else 'test'] = load_tiny_imagenet(train=train, device=device)
        train_ds, val_ds = self.__class__.DATASETS['train' if train else 'test'], None
        # FIX: train/validation split breaks things here
        val_size = kwargs.pop('val_size', None)
        use_val = False
        if train and use_val and val_size is not None:
            ts = len(train_ds)
            vs = int(ts * val_size) if type(val_size) == float and val_size < 1.0 else val_size
            ts -= vs
            train_ds, val_ds = torch.utils.data.random_split(train_ds, [ts, vs])
        self.ds = train_ds if not use_val or val_ds is None else val_ds
        self.extra_transforms = kwargs.pop('extra_transforms', None)
        # FIX: num_workers breaks things here
        kwargs.pop('num_workers', None)
        sampler = BatchSampler(RandomSampler(self.ds), batch_size=batch_size, drop_last=drop_last)
        super(TinyImagenetFastDataLoader, self).__init__(self.ds, sampler=sampler,
                                                         collate_fn=self.pre_process, **kwargs)
        self.train = train
        self.use_one_hot = use_one_hot

    @property
    def vis_transforms(self) -> Compose:
        return TinyImagenetDataset.vis_transforms()

    @property
    def n_classes(self) -> int:
        return 200

    def print_stats(self):
        title = "train" if self.train else "test"
        print(f"Number of labels in {title} dataset: {self.ds.features['label'].num_classes}")
        print(f"Number of examples in {title} dataset: {len(self.ds)}")
        print(f"Number of examples in {title} dataset: {len(self.ds)}")

    def pre_process(self, batch):
        x = batch[0]['image']
        y = batch[0]['label']

        # TODO: call self.extra_transforms to the PIL image (before being a tensor)
        #  probably it requires that the function load_tiny_imagenet() skips this line:
        #    ds.with_format("torch", device=device)

        if isinstance(x, list):
            # Not all examples in batch have the same number of color channels
            same_sizes = [
                example
                if len(example.shape) == 3 else
                example.unsqueeze(2).repeat(1, 1, 3)
                for example in x]

            x = torch.stack(same_sizes)

        x = x.permute(0, 3, 1, 2)
        x = layer.Normalize(dim=(0, 1, 2, 3))(x.float())
        return x, y if not self.use_one_hot else one_hot(y.clone(), self.n_classes).float()
