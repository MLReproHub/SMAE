import os
import tarfile
from typing import Tuple

import PIL
import pandas as pd
import torch
from PIL.Image import Image
from torch import Tensor
from torch.nn.functional import one_hot
from torch.utils.data import Dataset, DataLoader, BatchSampler, RandomSampler
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url, check_integrity

from utilities.path import data_path


def default_loader_rgb(path):
    return PIL.Image.open(path).convert('RGB')


def generate_transform_dict(origin_width: int = 64, width: int = 64, scale_ratio: float = 0.6) -> dict:
    """
    Source: https://github.com/bnu-wangxun/Deep_Metric/blob/master/DataSet/CUB200.py
    """
    normalize = transforms.Normalize(mean=[0.502, 0.459, 0.408], std=[0.5, 0.5, 0.5])
    return {
        'rand-crop': transforms.Compose([
            transforms.Resize(origin_width),
            transforms.RandomResizedCrop(scale=(scale_ratio, 1), size=width),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]),
        'random_crop': transforms.Compose([
            transforms.Resize(origin_width),
            transforms.RandomResizedCrop(scale=(scale_ratio, 1.0), size=width),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]),
        'center-crop': transforms.Compose([
            transforms.Resize(origin_width),
            transforms.CenterCrop(width),
            transforms.ToTensor(),
            normalize,
        ]),
        'center-crop-no-norm': transforms.Compose([
            transforms.Resize(origin_width),
            transforms.CenterCrop(width),
            transforms.ToTensor()
        ]),
        'resize': transforms.Compose([
            transforms.Resize(width),
            transforms.ToTensor(),
            normalize,
        ]),
        'auto_augment': transforms.Compose([
            transforms.Resize(origin_width),
            transforms.RandomResizedCrop(scale=(scale_ratio, 1.0), size=width),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            normalize,
        ]),
    }


class CubDataset(Dataset):
    ZIP_URL = 'https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1'
    ZIP_FNAME = 'CUB_200_2011.tgz'
    ZIP_MD5 = '97eceeb196236b17998738112f37df78'

    IMG_WIDTH = 64
    TRANSFORMS = generate_transform_dict(origin_width=IMG_WIDTH, width=IMG_WIDTH)

    def __init__(self, train: bool = True, transforms_key: str = 'center-crop', load_bboxes: bool = False,
                 use_one_hot: bool = False):
        self.data_path = os.path.join(data_path, 'cub-200-2011')
        self.img_dir_path = os.path.join(self.data_path, 'CUB_200_2011/images')
        self.transforms = self.__class__.TRANSFORMS[transforms_key]
        self.train = train

        self._ensure_data_exist()
        # Data Columns: ['img_id', 'filepath', 'target', 'target_name', 'is_training_img', (optional) 'bbox.{x,y,w,h}']
        self.data: pd.DataFrame = self._load(load_bboxes=load_bboxes)
        self.load_bboxes = load_bboxes
        self.bbox_cols = [c for c in self.data if c.startswith('bbox.')]
        self.use_one_hot = use_one_hot

    def _ensure_data_exist(self):
        os.makedirs(self.data_path, exist_ok=True)
        if not os.path.exists(self.img_dir_path):
            # Download
            zip_fpath = os.path.join(self.data_path, self.__class__.ZIP_FNAME)
            if not os.path.exists(zip_fpath) or not check_integrity(zip_fpath):
                download_url(self.__class__.ZIP_URL, self.data_path, self.__class__.ZIP_FNAME, self.__class__.ZIP_MD5)
            # Unzip
            with tarfile.open(zip_fpath, "r:gz") as tar:
                print(f'\t[{self.__class__.__name__}::_ensure_data_exist] Extracting "{zip_fpath}"')
                tar.extractall(path=self.data_path)

    def _load(self, load_bboxes: bool = False):
        images = pd.read_csv(os.path.join(self.data_path, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.data_path, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.data_path, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])
        class_names = pd.read_csv(os.path.join(self.data_path, 'CUB_200_2011', 'classes.txt'),
                                  sep=' ', names=['target', 'target_name'])
        data = images.merge(image_class_labels.merge(class_names, on='target'), on='img_id')
        data = data.merge(train_test_split, on='img_id')
        if load_bboxes:
            data = data.merge(
                pd.read_csv(os.path.join(self.data_path, 'CUB_200_2011', 'bounding_boxes.txt'),
                            sep=' ', names=['img_id', 'bbox.x', 'bbox.y', 'bbox.w', 'bbox.h']),
                on='img_id')
        return data[data.is_training_img == (1 if self.train else 0)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Image or Tensor, int] or Tuple[Image or Tensor, int, Tensor]:
        sample = self.data.iloc[idx]
        path = os.path.join(self.img_dir_path, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        target = torch.tensor(target) if not self.use_one_hot else one_hot(torch.tensor(target), 200)
        img = default_loader_rgb(path)
        if self.transforms is not None:
            img = self.transforms(img)
        if self.load_bboxes:
            return img, target.float(), torch.from_numpy(sample[self.bbox_cols].astype(float).to_numpy())
        return img, target.float()


class CubSegmentationDataset(CubDataset):
    SEGM_ZIP_URL = 'https://data.caltech.edu/records/w9d68-gec53/files/segmentations.tgz?download=1'
    SEGM_ZIP_FNAME = 'segmentations.tgz'
    SEGM_ZIP_MD5 = '4d47ba1228eae64f2fa547c47bc65255'

    def __init__(self, segm_transforms_key: str = 'center-crop-no-norm', *cub_args, **cub_kwargs):
        super(CubSegmentationDataset, self).__init__(*cub_args, **cub_kwargs)
        self.segm_dir_path = os.path.join(self.data_path, 'CUB_200_2011/segmentations')
        self._ensure_segmentation_data_exit()
        self.segm_transforms = self.__class__.TRANSFORMS[segm_transforms_key]

    def _ensure_segmentation_data_exit(self):
        if not os.path.exists(self.segm_dir_path):
            # Download
            zip_fpath = os.path.join(self.data_path, self.__class__.SEGM_ZIP_FNAME)
            if not os.path.exists(zip_fpath) or not check_integrity(zip_fpath):
                download_url(self.__class__.SEGM_ZIP_URL, self.data_path, self.__class__.SEGM_ZIP_FNAME,
                             self.__class__.SEGM_ZIP_MD5)
            # Unzip
            with tarfile.open(zip_fpath, "r:gz") as tar:
                print(f'\t[{self.__class__.__name__}::_ensure_data_exist] Extracting "{zip_fpath}"')
                tar.extractall(path=os.path.dirname(self.segm_dir_path))

    def __getitem__(self, idx: int) -> Tuple[Image or Tensor, int, Tensor] or \
                                       Tuple[Image or Tensor, int, Tensor, Tensor]:
        img, target, bbox = super(CubSegmentationDataset, self).__getitem__(idx)
        segm_fpath = os.path.join(self.segm_dir_path, self.data.iloc[idx].filepath.replace(".jpg", ".png"))
        seg = default_loader(segm_fpath)
        if self.segm_transforms is not None:
            seg = self.segm_transforms(seg)
        if self.load_bboxes:
            return img, target, bbox, seg
        return img, target, seg  # seg is a 0-1 mask of the same shape as img


class CubDataLoader(DataLoader):
    """
    CubDataLoader Class:
    This class is used to access CUB-200-2011 dataset via PyTorch's Dataloading API.
    """

    def __init__(self, train=True, ds_transforms_key: str = 'center-crop', device: str = 'cpu', use_val: bool = False,
                 val_size=None, use_one_hot: bool = True, **kwargs):
        self.use_one_hot = use_one_hot
        train_ds, val_ds = CubDataset(train=train, transforms_key=ds_transforms_key, use_one_hot=use_one_hot), None
        if train and use_val and val_size is not None:
            ts = len(train_ds)
            vs = int(ts * val_size) if type(val_size) == float and val_size < 1.0 else val_size
            ts -= vs
            train_ds, val_ds = torch.utils.data.random_split(train_ds, [ts, vs])
        ds = train_ds if not use_val or val_ds is None else val_ds
        if 'pin_memory' not in kwargs:
            kwargs['pin_memory'] = device != 'cpu'
        super(CubDataLoader, self).__init__(dataset=ds, shuffle=True, **kwargs)

    @property
    def vis_transforms(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.Normalize(mean=[0., 0., 0.], std=[2.0, 2.0, 2.0]),
            transforms.Normalize(mean=[-0.502, -0.459, -0.408], std=[1.0, 1.0, 1.0])
        ])

    @property
    def n_classes(self) -> int:
        # noinspection PyUnresolvedReferences
        return 200


if __name__ == '__main__':
    # ds_ = CubDataset(load_bboxes=True)
    # print(ds_.data.columns)
    # print(ds_.data.iloc[100])
    #
    # ds_ = CubSegmentationDataset(load_bboxes=True)
    # print(ds_.data.columns)
    # print(ds_.data.iloc[100])
    # ds_100_ = ds_[100]
    # print(ds_100_[0].shape, ds_100_[3].shape)
    # print(ds_100_[3].min(), ds_100_[3].max())

    # Dataloader
    dl_ = CubDataLoader(train=True, batch_size=1, device='cpu')
    batch_ = next(iter(dl_))
    print(batch_[0].shape)

    import torchvision.transforms.functional as F
    import matplotlib.pyplot as plt

    # plt.imshow(F.to_pil_image(batch_[0][0]))
    plt.imshow(F.to_pil_image(dl_.vis_transforms(batch_[0][0])))
    plt.show()
