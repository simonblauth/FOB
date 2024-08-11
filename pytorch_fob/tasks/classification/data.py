from pathlib import Path
from typing import Callable, Optional, Sequence
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


class ImageFolderCached(ImageFolder):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.images = []
        self.labels = []

        log_info("loading dataset to memory...")
        for path, label in self.samples:
            self.images.append(self.loader(path))
            self.labels.append(label)

    def __getitem__(self, index):
        image = self.images[index] if self.transform is None else self.transform(self.images[index])
        label = self.targets[index] if self.target_transform is None else self.target_transform(self.targets[index])
        return image, label


class Imagenet64Dataset(Dataset):
    def __init__(self, data_source, transform) -> None:
        super().__init__()
        self.data = data_source
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        img = np.array(self.data[index]["image"])  # need to make copy because original is not writable
        return self.transform(img), self.data[index]["label"]


class Imagenet64DatasetCached(Imagenet64Dataset):
    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.images = []
        self.labels = []

        log_info("loading dataset to memory...")
        for item in self.data:
            self.images.append(np.array(item["image"]))
            self.labels.append(np.array(item["label"]))

    def __getitem__(self, index):
        return self.transform(self.images[index]), self.labels[index]


class ImagenetDataModule(TaskDataModule):
    def __init__(self, config: TaskConfig):
        super().__init__(config)
        self.image_size = config.image_size
        self.resize = self.image_size not in [16, 32, 64]
        self.train_transforms = self._get_train_transforms(config)
        self.val_transforms = self._get_val_transforms()

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

    def _load_dataset(
            self,
            split: Optional[str],
            cache_data: bool = False,
            download: bool = False
        ) -> Dataset:
        if self.image_size in [16, 32, 64]:
            ds = tfds.data_source(
                f"imagenet_resized/{self.image_size}x{self.image_size}",
                split=split,
                data_dir=self.data_dir,
                download=download
            )
            ds_cls = Imagenet64DatasetCached if cache_data else Imagenet64Dataset
        elif self.image_size == 224:
            path = self.data_dir / "imagenet_full" if "imagenet_path" not in self.config else Path(self.config.imagenet_path)
            if download:
                _check_imagenet_files(path)
            if split == "train" or split is None:
                ds = str(path / "train")
            elif split == "validation":
                ds = str(path / "val")
            else:
                raise ValueError(f"Unknown split {split}")
            ds_cls = ImageFolderCached if cache_data else ImageFolder
        else:
            raise ValueError(f"Image size {self.image_size} not supported")
        return ds_cls(ds, transform=self._get_transforms_from_split(split))  # type: ignore
        
    def _get_transforms_from_split(self, split: Optional[str]):
        if split is None:
            return None
        if split == "train":
            return self.train_transforms
        elif split == "validation":
            return self.val_transforms
        else:
            raise ValueError(f"Unknown split {split}")


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
