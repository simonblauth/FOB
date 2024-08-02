import importlib
from typing import Any
from pathlib import Path
from lightning import LightningModule, LightningDataModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import nn
from torch.utils.data import DataLoader, Dataset
from pytorch_fob.engine.utils import log_info
from pytorch_fob.optimizers import Optimizer
from pytorch_fob.engine.configs import TaskConfig
from pytorch_fob.engine.parameter_groups import GroupedModel


def import_task(name: str):
    return importlib.import_module(f"pytorch_fob.tasks.{name}.task")


def task_path(name: str) -> Path:
    return Path(__file__).resolve().parent / name


def task_names() -> list[str]:
    EXCLUDE = ["__pycache__"]
    return [d.name for d in Path(__file__).parent.iterdir() if d.is_dir() and d.name not in EXCLUDE]


class TaskModel(LightningModule):
    def __init__(
            self,
            model: nn.Module | GroupedModel,
            optimizer: Optimizer,
            config: TaskConfig,
            **kwargs: Any
            ) -> None:
        super().__init__(**kwargs)
        self.config = config
        self.optimizer = optimizer
        self.model = model if isinstance(model, GroupedModel) else GroupedModel(model)

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return self.optimizer.configure_optimizers(self.model)


class TaskDataModule(LightningDataModule):
    def __init__(self, config: TaskConfig) -> None:
        super().__init__()
        self.config = config
        self.workers: int = min(config.workers, 16)
        self.data_dir: Path = config.data_dir / config.name
        self.batch_size: int = config.batch_size
        self.data_train: Any
        self.data_val: Any
        self.data_test: Any
        self.data_predict: Any
        self.collate_fn = None

    def check_dataset(self, data):
        """Make sure that all tasks have correctly configured their data sets"""
        if not data:
            raise NotImplementedError("Each task has its own data set")
        if not self.batch_size or self.batch_size < 1:
            raise NotImplementedError("Each task configures its own batch_size. \
                                      Please set it explicitely, to avoid confusion.")

    def cache_data(self, stage: str):
        if not self.config.cache_data:
            return
        if stage == "fit":
            self._cache_dataset("train", self.data_train)
            self._cache_dataset("val", self.data_val)
        if stage == "validate":
            self._cache_dataset("val", self.data_val)
        if stage == "test":
            self._cache_dataset("test", self.data_test)
        if stage == "predict":
            self._cache_dataset("predict", self.data_predict)

    def _cache_dataset(self, dataset: str, data: Dataset):
        log_info(f"loading {dataset} dataset to memory...")
        data = CachedDataset(data)

    def train_dataloader(self):
        self.check_dataset(self.data_train)
        return DataLoader(
            self.data_train,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.workers,
            collate_fn=self.collate_fn
        )

    def val_dataloader(self):
        self.check_dataset(self.data_val)
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            num_workers=self.workers,
            collate_fn=self.collate_fn
        )

    def test_dataloader(self):
        self.check_dataset(self.data_test)
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            num_workers=self.workers,
            collate_fn=self.collate_fn
        )

    def predict_dataloader(self):
        self.check_dataset(self.data_predict)
        return DataLoader(
            self.data_predict,
            batch_size=self.batch_size,
            num_workers=self.workers,
            collate_fn=self.collate_fn
        )


class CachedDataset(Dataset):
    def __init__(self, data: Dataset):
        self.cache = []
        for i in data:
            self.cache.append(i)

    def __getitem__(self, index):
        return self.cache[index]

    def __len__(self):
        return len(self.cache)
