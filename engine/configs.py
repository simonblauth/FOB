from pathlib import Path
from typing import Any, Optional
from .utils import AttributeDict, convert_type_inside_dict


class BaseConfig(AttributeDict):
    def __init__(self, config: dict):
        super().__init__(convert_type_inside_dict(config, dict, AttributeDict))


class NamedConfig(BaseConfig):
    def __init__(
            self,
            config: dict[str, Any],
            identifier_key: str = "name",
            outdir_key: str = "output_dir_name"
            ) -> None:
        super().__init__(config)
        self.name = config[identifier_key]
        self.output_dir_name = config[outdir_key]


class OptimizerConfig(NamedConfig):
    def __init__(
            self,
            config: dict[str, Any],
            optimizer_key: str,
            task_key: str,
            identifier_key: str = "name",
            outdir_key: str = "output_dir_name"
            ) -> None:
        cfg = dict(config[optimizer_key])
        self.lr_interval: Literal["step", "epoch"] = cfg.get("lr_interval", "step")
        self.max_steps = config[task_key]["max_steps"]
        cfg["max_steps"] = self.max_steps
        super().__init__(cfg, identifier_key, outdir_key)


class TaskConfig(NamedConfig):
    def __init__(
            self,
            config: dict[str, Any],
            task_key: str,
            engine_key: str,
            identifier_key: str = "name",
            outdir_key: str = "output_dir_name"
            ) -> None:
        cfg = dict(config[task_key])
        self.batch_size: int = cfg["batch_size"]
        self.data_dir = Path(config[engine_key]["data_dir"]).resolve()
        self.max_epochs: int = cfg["max_epochs"]
        self.max_steps: int = cfg["max_steps"]
        self.target_metric: str = cfg["target_metric"]
        self.target_metric_mode: str = cfg["target_metric_mode"]
        self.workers = config[engine_key]["workers"]
        cfg["data_dir"] = self.data_dir
        cfg["workers"] = self.workers
        super().__init__(cfg, identifier_key, outdir_key)


class EngineConfig(BaseConfig):
    def __init__(self, config: dict[str, Any], task_key: str, engine_key: str) -> None:
        cfg = dict(config[engine_key])
        self.deterministic: bool = cfg.get("deterministic", True)
        self.devices: int = cfg["devices"]
        self.data_dir = Path(cfg["data_dir"]).resolve()
        self.log_extra: bool = cfg.get("log_extra", False)
        self.max_steps = config[task_key]["max_steps"]
        self.optimize_memory: bool = cfg.get("optimize_memory", False)
        self.output_dir = Path(cfg["output_dir"]).resolve()
        maybe_resume = cfg.get("resume", None)
        self.resume: Optional[Path] = Path(maybe_resume).resolve() if maybe_resume is not None else None
        self.seed: int = cfg["seed"]
        self.seed_mode: str = cfg["seed_mode"]
        self.silent: bool = cfg.get("silent", False)
        self.test_only: bool = cfg.get("test_only", False)
        self.workers: int = cfg["workers"]
        cfg["max_steps"] = self.max_steps
        super().__init__(cfg)
