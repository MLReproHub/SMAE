from unittest import TestCase

from tqdm import tqdm

from dataset.tiny_imagenet import TinyImagenetDataLoader, TinyImagenetFastDataLoader


class TestTinyImagenetDataLoader(TestCase):
    def setUp(self) -> None:
        self.batch_size = 16
        self.use_oh = 16
        self.dataloader = TinyImagenetDataLoader(train=False, use_one_hot=self.use_oh, batch_size=self.batch_size,
                                                 drop_last=True)

    def test_shapes(self):
        for x, y in tqdm(self.dataloader):
            self.assertListEqual(list(x.shape), [self.batch_size, 3, 64, 64])
            self.assertListEqual(list(y.shape), [self.batch_size, 200] if self.use_oh else [self.batch_size, ])


class TestTinyImagenetFastDataLoader(TestCase):
    def setUp(self) -> None:
        self.batch_size = 16
        self.dataloader = TinyImagenetFastDataLoader(train=False, batch_size=self.batch_size, drop_last=True,
                                                     device='cpu')

    def test_shapes(self):
        for x in tqdm(self.dataloader):
            self.assertListEqual(list(x.shape), [self.batch_size, 3, 64, 64])
