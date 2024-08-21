from pathlib import Path
from typing import Callable, Optional, Sequence
import numpy as np
import tensorflow_datasets as tfds
import webdataset as wds
import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageNet
from torchvision.transforms import v2
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, create_transform
from pytorch_fob.engine.utils import log_info
from pytorch_fob.tasks import TaskDataModule
from pytorch_fob.engine.configs import TaskConfig


class RepeatedImageNet(ImageNet):
    def __init__(self, *args, num_augmentations: int = 1, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_augmentations = num_augmentations

    def __len__(self) -> int:
        return len(self.samples) * self.num_augmentations

    def __getitem__(self, index):
        orig_index = index // self.num_augmentations
        return super().__getitem__(orig_index)


class Imagenet64Dataset(Dataset):
    def __init__(self, data_source, transform, num_augmentations: int = 1) -> None:
        super().__init__()
        self.data = data_source
        self.transform = transform
        self.num_augmentations = num_augmentations

    def __len__(self) -> int:
        return len(self.data) * self.num_augmentations

    def __getitem__(self, index):
        orig_index = index // self.num_augmentations
        img = np.array(self.data[orig_index]["image"])  # need to make copy because original is not writable
        return self.transform(img), self.data[orig_index]["label"]


class Imagenet64DatasetCached(Imagenet64Dataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.images = []
        self.labels = []

        log_info("loading dataset to memory...")
        for item in self.data:
            self.images.append(np.array(item["image"]))
            self.labels.append(np.array(item["label"]))

    def __getitem__(self, index):
        orig_index = index // self.num_augmentations
        return self.transform(self.images[orig_index]), self.labels[orig_index]


class ImagenetDataModule(TaskDataModule):
    def __init__(self, config: TaskConfig):
        super().__init__(config)
        self.image_size = config.image_size
        self.resize = self.image_size not in [16, 32, 64]
        self.train_transforms = self._get_train_transforms(config)
        self.val_transforms = self._get_val_transforms()
        self.num_augmentations = config.train_transforms.repeat_augmentations
        self.shardsize = 10_000
        if self.resize:
            # no collate fn needed for wds
            self.collate_fn = identity 

    def _get_transforms(self, extra: Sequence[Callable] = tuple()):
        return v2.Compose([
            v2.ToImage(),
            *self._get_resized_transforms(),
            *extra,
            v2.ToDtype(torch.float, scale=True),
            v2.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            v2.ToPureTensor()
        ])

    def _get_resized_transforms(self):
        if self.resize:
            return [
                v2.Resize(
                    self.image_size,
                    interpolation=v2.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                v2.CenterCrop(self.image_size),
            ]
        return []

    def _get_val_transforms(self):
        return self._get_transforms()

    def _get_train_transforms(self, config: TaskConfig):
        # override for timm transforms
        if "kwargs_timm" in config.train_transforms:
            tfs = create_transform(
                input_size=self.image_size,
                is_training=True,
                **config.train_transforms.kwargs_timm
            )
            tfs.transforms.insert(0, v2.ToPILImage())
            return tfs
        # reading settings
        tfs = []
        if config.train_transforms.random_crop.use:
            random_crop = v2.RandomCrop(
                size=config.train_transforms.random_crop.size,
                padding=config.train_transforms.random_crop.padding,
                padding_mode=config.train_transforms.random_crop.padding_mode
            )
            tfs.append(random_crop)
        if config.train_transforms.horizontal_flip.use:
            horizontal_flip = v2.RandomHorizontalFlip(config.train_transforms.horizontal_flip.p)
            tfs.append(horizontal_flip)
        if config.train_transforms.trivial_augment.use:
            trivial_augment = v2.TrivialAugmentWide(interpolation=v2.InterpolationMode.BILINEAR)
            tfs.append(trivial_augment)

        transforms = self._get_transforms(tfs)

        return transforms

    def prepare_data(self):
        # download
        # TODO: find a solution to remove tensorflow from requirements as it is only needed for the download
        self._load_dataset(split=None, cache_data=False, download=True)

    def setup(self, stage: str):
        """setup is called from every process across all the nodes. Setting state here is recommended.
        """
        self._setup(stage)

    def _setup(self, stage: str, cache_data: bool = False):
        if stage == "fit":
            self.data_train = self._load_dataset("train", cache_data=cache_data)
            self.data_val = self._load_dataset("validation", cache_data=cache_data)
        if stage == "validate":
            self.data_val = self._load_dataset("validation", cache_data=cache_data)
        if stage == "test":
            self.data_test = self._load_dataset("validation", cache_data=cache_data)
        if stage == "predict":
            self.data_predict = self._load_dataset("validation", cache_data=cache_data)

    def cache_data(self, stage: str):
        if not self.config.cache_data:
            return
        self._setup(stage, cache_data=True)

    def train_dataloader(self):
        if isinstance(self.data_train, wds.DataPipeline):
            return wds.WebLoader(
                self.data_train,
                batch_size=None,
                shuffle=False,
                num_workers=self.workers,
                collate_fn=self.collate_fn,
            ).unbatched().shuffle(self.shardsize).batched(self.batch_size).with_epoch(self._wds_epoch("train")).with_length(self._wds_epoch("train"))
        return super().train_dataloader()

    def val_dataloader(self):
        if isinstance(self.data_val, wds.DataPipeline):
            return wds.WebLoader(
                self.data_val,
                batch_size=None,
                shuffle=False,
                num_workers=self.workers,
                collate_fn=self.collate_fn,
            ).with_length(self._wds_epoch("validation"))
        return super().val_dataloader()

    def test_dataloader(self):
        if isinstance(self.data_test, wds.DataPipeline):
            return wds.WebLoader(
                self.data_test,
                batch_size=None,
                shuffle=False,
                num_workers=self.workers,
                collate_fn=self.collate_fn,
            ).with_length(self._wds_epoch("test", devices=1))
        return super().test_dataloader()

    def predict_dataloader(self):
        if isinstance(self.data_predict, wds.DataPipeline):
            return wds.WebLoader(
                self.data_predict,
                batch_size=None,
                shuffle=False,
                num_workers=self.workers,
                collate_fn=self.collate_fn,
            ).with_length(self._wds_epoch("predict"))
        return super().predict_dataloader()

    def _load_dataset(
            self,
            split: Optional[str],
            cache_data: bool = False,
            download: bool = False
        ) -> Dataset:
        num_augmentations = self.num_augmentations if split == "train" else 1
        # resized imagenet
        if self.image_size in [16, 32, 64]:
            dsrc = tfds.data_source(
                f"imagenet_resized/{self.image_size}x{self.image_size}",
                split=split,
                data_dir=self.data_dir,
                download=download
            )
            ds_cls = Imagenet64DatasetCached if cache_data else Imagenet64Dataset
            return ds_cls(
                dsrc,
                transform=self._get_transforms_from_split(split),
                num_augmentations=num_augmentations,
            )
        # full imagenet
        path = self.data_dir / "imagenet_full" if "imagenet_path" not in self.config else Path(self.config.imagenet_path)
        if (path / "shards").is_dir():
            urls = str(path / "shards" / self._wds_shard_pattern(split))
            sl = wds.ResampledShards if split == "train" else wds.SimpleShardList
            shard_shuffle = self._wds_shard_count(split) if split == "train" else 0
            sample_shuffle = self.shardsize if split == "train" else 0
            return wds.DataPipeline(
                sl(urls),
                wds.shuffle(shard_shuffle),
                wds.split_by_node,
                wds.split_by_worker,
                wds.tarfile_to_samples(),
                wds.shuffle(sample_shuffle),
                wds.decode("pil"),
                wds.to_tuple("jpg", "cls"),
                wds.map_tuple(self._get_transforms_from_split(split), torch.tensor),
                wds.batched(self.batch_size, partial=False),
            ).with_length(self._wds_len(split))
        return RepeatedImageNet(
            path,
            split=split.replace("validation", "val"),
            transform=self._get_transforms_from_split(split),
            num_augmentations=num_augmentations,
        )

    def _wds_shard_count(self, split: Optional[str]):
        if split is None or split == "train":
            return 147
        else:
            return 7

    def _wds_shard_pattern(self, split: Optional[str]):
        s = "train" if split is None or split == "train" else "val"
        lo, hi = 0, self._wds_shard_count(split) - 1
        return f"imagenet-{s}-{{{lo:06d}..{hi:06d}}}.tar"

    def _wds_len(self, split: Optional[str] = None):
        aug = self.num_augmentations if split is None or split == "train" else 1
        return aug * self._wds_shard_count(split) * self.shardsize

    def _wds_epoch(self, split: Optional[str] = None, devices: Optional[int] = None):
        dev = self.config.devices if devices is None else devices
        return self._wds_len(split) // self.batch_size // dev

    def _get_transforms_from_split(self, split: Optional[str]):
        if split is None:
            return None
        if split == "train":
            return self.train_transforms
        elif split == "validation":
            return self.val_transforms
        else:
            raise ValueError(f"Unknown split {split}")


def identity(x):
    return x
