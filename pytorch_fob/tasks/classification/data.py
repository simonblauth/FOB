from pathlib import Path
from typing import Any, Callable, Optional, Sequence
import numpy as np
import tensorflow_datasets as tfds
import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import ImageFolder
from torchvision.transforms import v2
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, create_transform
from pytorch_fob.engine.utils import log_info
from pytorch_fob.tasks import TaskDataModule
from pytorch_fob.engine.configs import TaskConfig

class Imagenet64Dataset(Dataset):
    def __init__(self, data_source) -> None:
        super().__init__()
        self.data = data_source
        self.transforms: Any

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        img = np.array(self.data[index]["image"])  # need to make copy because original is not writable
        return {"image": self.transforms(img), "label": self.data[index]["label"]}

    def set_transform(self, transforms):
        self.transforms = transforms


class Imagenet64DatasetCached(Imagenet64Dataset):
    def __init__(self, data_source) -> None:
        super().__init__(data_source)
        self.images = []
        self.labels = []

        log_info("loading dataset to memory...")
        for item in self.data:
            self.images.append(np.array(item["image"]))
            self.labels.append(np.array(item["label"]))

    def __getitem__(self, index):
        return {"image": self.transforms(self.images[index]), "label": self.labels[index]}


class ImagenetDataModule(TaskDataModule):
    def __init__(self, config: TaskConfig):
        super().__init__(config)
        self.image_size = config.image_size
        self.train_transforms = self._get_train_transforms(config)
        self.val_transforms = self._get_transforms()

    def _get_transforms(self, extra: Sequence[Callable] = tuple()):
        return v2.Compose([
            v2.ToImage(),
            *extra,
            v2.ToDtype(torch.float, scale=True),
            v2.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            v2.ToPureTensor()
        ])

    def _get_train_transforms(self, config: TaskConfig):
        # override for timm transforms
        if "kwargs_timm" in config.train_transforms:
            return create_transform(
                input_size=self.image_size,
                is_training=True,
                **config.train_transforms.timm_kwargs
            )
        # reading setting
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
            self.data_train.set_transform(self.train_transforms)
            self.data_val.set_transform(self.val_transforms)
        if stage == "validate":
            self.data_val = self._load_dataset("validation", cache_data=cache_data)
            self.data_val.set_transform(self.val_transforms)
        if stage == "test":
            self.data_test = self._load_dataset("validation", cache_data=cache_data)
            self.data_test.set_transform(self.val_transforms)
        if stage == "predict":
            self.data_predict = self._load_dataset("validation", cache_data=cache_data)
            self.data_predict.set_transform(self.val_transforms)

    def cache_data(self, stage: str):
        if not self.config.cache_data:
            return
        self._setup(stage, cache_data=True)

    def _load_dataset(
            self,
            split: Optional[str],
            cache_data: bool = False,
            download: bool = False
        ) -> Imagenet64Dataset:
        if self.image_size in [16, 32, 64]:
            ds = tfds.data_source(
                "imagenet_resized/64x64",
                split=split,
                data_dir=self.data_dir,
                download=download
            )
        elif self.image_size == 224:
            path = self.data_dir / "imagenet_full" if "imagenet_path" not in self.config else Path(self.config.imagenet_path)
            if download:
                _check_imagenet_files(path)
            if split == "train":
                ds = ImageFolder(str(path / "train"), transform=self.train_transforms)  # type: ignore
            elif split == "validation":
                ds = ImageFolder(str(path / "val"), transform=self.val_transforms)
        else:
            raise ValueError(f"Image size {self.image_size} not supported")
        rds = Imagenet64DatasetCached(ds) if cache_data else Imagenet64Dataset(ds)
        return rds


def _check_imagenet_files(data_dir: Path):
    valid = data_dir.is_dir()
    msg = f"Please download the full imagenet dataset to {data_dir} or specify a different path under `task.imagenet_path`"
    if not valid:
        raise ValueError(msg)
    def count_images_in_directory(d: Path) -> int:
        return sum(1 for _ in d.rglob('*.[Jj][Pp]*[Gg]'))
    valid = valid and count_images_in_directory(data_dir / "train") == 1281167
    valid = valid and count_images_in_directory(data_dir / "val") == 50000
    if not valid:
        raise ValueError(msg)
