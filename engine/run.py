import hashlib
from pathlib import Path
import time
from typing import Any, Optional

from lightning import Callback, LightningDataModule, LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import Logger, TensorBoardLogger, CSVLogger
from lightning_utilities.core.rank_zero import rank_zero_info, rank_zero_warn
import torch
import yaml
from engine.callbacks import LogParamsAndGrads, PrintEpoch
from engine.configs import EngineConfig, EvalConfig, OptimizerConfig, TaskConfig
from engine.utils import AttributeDict, EndlessList, calculate_steps, concatenate_dict_keys, convert_type_inside_dict, dict_differences, findfirst, path_to_str_inside_dict, precision_with_fallback, seconds_to_str, trainer_strategy, write_results
from optimizers.optimizers import Optimizer
from tasks.tasks import TaskDataModule, TaskModel, import_task

class Run():
    def __init__(
            self,
            config: dict[str, Any],
            default_config: dict[str, Any],
            task_key: str,
            optimizer_key: str,
            engine_key: str,
            eval_key: str,
            identifier_key: str
            ) -> None:
        self._config = config
        self._default_config = default_config
        self.task_key = task_key
        self.optimizer_key = optimizer_key
        self.engine_key = engine_key
        self.eval_key = eval_key
        self.identifier_key = identifier_key
        self._generate_configs()
        self._set_outpath()
        self._callbacks = AttributeDict({})

    def start(self):
        self._ensure_max_steps()
        self._ensure_resume_path()
        torch.set_float32_matmul_precision('high')
        seed_everything(self.engine.seed, workers=True)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.export_config()
        model, data_module = self.get_task()
        # TODO: test only correctness for last checkpoint
        if self.engine.train:
            trainer = self.get_trainer()
            self.train(trainer, model, data_module)
        if self.engine.test:
            tester = self.get_tester()
            self.test(tester, model, data_module)
            best_path = self.get_best_checkpoint()
            if best_path is not None:
                self.test(tester, model, data_module, Path(best_path))

    def train(self, trainer: Trainer, model: LightningModule, data_module: LightningDataModule):
        start_time = time.time()
        if self.engine.accelerator == "gpu" and torch.cuda.is_available():
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True,
                enable_math=True,
                enable_mem_efficient=(self.engine.optimize_memory or not self.engine.deterministic)
            ):
                trainer.fit(model, datamodule=data_module, ckpt_path=self.engine.resume)  # type: ignore
        else:
            trainer.fit(model, datamodule=data_module, ckpt_path=self.engine.resume)  # type: ignore
        end_time = time.time()
        train_time = int(end_time - start_time)
        rank_zero_info(f"Finished training in {seconds_to_str(train_time)}.")

    def test(self, tester: Trainer, model: LightningModule, data_module: LightningDataModule, ckpt: Optional[Path] = None):
        ckpt_path = self.engine.resume if ckpt is None else ckpt
        mode = "final" if ckpt_path is None or ckpt_path.stem.startswith("last") else "best"  # type: ignore
        score = tester.test(model, datamodule=data_module, ckpt_path=ckpt_path)  # type: ignore
        write_results(score, self.run_dir / f"results_{mode}_model.json")

    def export_config(self):
        with open(self.run_dir / "config.yaml", "w", encoding="utf8") as f:
            d = path_to_str_inside_dict(self._config)
            d = convert_type_inside_dict(d, EndlessList, list)
            yaml.safe_dump(d, f)

    def get_config(self) -> AttributeDict:
        return AttributeDict(self._config)

    def get_optimizer(self) -> Optimizer:
        return Optimizer(self.optimizer)

    def get_task(self) -> tuple[TaskModel, TaskDataModule]:
        task_module = import_task(self.task.name)
        return task_module.get_task(self.get_optimizer(), self.task)

    def get_datamodule(self) -> TaskDataModule:
        task_module = import_task(self.task.name)
        return task_module.get_datamodule(self.task)

    def get_callbacks(self) -> list[Callback]:
        if len(self._callbacks) < 1:
            self._init_callbacks()
        return list(self._callbacks.values())

    def get_loggers(self) -> list[Logger]:
        return [
            TensorBoardLogger(
                save_dir=self.run_dir,
                name="tb_logs"
            ),
            CSVLogger(
                save_dir=self.run_dir,
                name="csv_logs"
            )
        ]

    def get_trainer(self) -> Trainer:
        return Trainer(
            max_steps=self.engine.max_steps,
            logger=self.get_loggers(),
            callbacks=self.get_callbacks(),
            devices=self.engine.devices,
            strategy=trainer_strategy(self.engine.devices),
            enable_progress_bar=(not self.engine.silent),
            deterministic=self.engine.deterministic,
            detect_anomaly=self.engine.detect_anomaly,
            gradient_clip_val=self.engine.gradient_clip_val,
            gradient_clip_algorithm=self.engine.gradient_clip_alg,
            precision=precision_with_fallback(self.engine.precision),  # type: ignore
            accelerator=self.engine.accelerator
        )

    def get_tester(self) -> Trainer:
        return Trainer(
            devices=1,
            enable_progress_bar=(not self.engine.silent),
            deterministic=self.engine.deterministic,
            precision=precision_with_fallback(self.engine.precision),  # type: ignore
            accelerator=self.engine.accelerator
        )

    def get_best_checkpoint(self) -> Optional[Path]:
        model_checkpoint = self._callbacks.get("model_checkpoint", None)
        if model_checkpoint is not None:
            model_checkpoint = model_checkpoint.best_model_path
        if model_checkpoint is None:
            available_checkpoints = self.get_available_checkpoints()
            model_checkpoint = findfirst(lambda x: x.stem.startswith("best"), available_checkpoints)
        return model_checkpoint

    def get_available_checkpoints(self) -> list[Path]:
        if self.checkpoint_dir.exists():
            return list(filter(lambda x: x.suffix == ".ckpt", self.checkpoint_dir.iterdir()))
        return []

    def _ensure_resume_path(self):
        if isinstance(self.engine.resume, Path):
            pass
        elif isinstance(self.engine.resume, bool):
            resume_path = None
            if self.engine.resume:
                available_checkpoints = self.get_available_checkpoints()
                if len(available_checkpoints) < 1:
                    rank_zero_warn("engine.resume=True but no checkpoint was found. Starting run from scratch.")
                else:
                    resume_path = findfirst(lambda x: x.stem == "last", available_checkpoints)
            self._config[self.engine_key]["resume"] = resume_path
            self._generate_configs()
        else:
            raise TypeError(f"Unsupportet type for 'resume', got {type(self.engine.resume)=}.")

    def _ensure_max_steps(self):
        if self.task.max_steps is None:
            max_steps = self._calc_max_steps()
            self._config[self.task_key]["max_steps"] = max_steps
            if self._default_config[self.task_key]["max_steps"] is None:
                self._default_config[self.task_key]["max_steps"] = max_steps
            self._generate_configs()
            rank_zero_info(f"'max_steps' not set explicitly, using {max_steps=} (calculated from " +
            f"max_epochs={self.task.max_epochs}, batch_size={self.task.batch_size}, devices={self.engine.devices})")

    def _calc_max_steps(self) -> int:
        dm = self.get_datamodule()
        dm.setup("fit")
        train_samples = len(dm.data_train)
        return calculate_steps(self.task.max_epochs, train_samples, self.engine.devices, self.task.batch_size)

    def _init_callbacks(self):
        self._callbacks["model_checkpoint"] = ModelCheckpoint(
            dirpath=self.checkpoint_dir,
            filename="best-{epoch}-{step}",
            monitor=self.task.target_metric,
            mode=self.task.target_metric_mode,
            save_last=True
        )
        self._callbacks["early_stopping"] = EarlyStopping(
            monitor=self.task.target_metric,
            mode=self.task.target_metric_mode,
            patience=self.engine.early_stopping,
            check_finite=self.engine.check_finite,
            log_rank_zero_only=True
        )
        self._callbacks["lr_monitor"] = LearningRateMonitor(logging_interval=self.optimizer.lr_interval)
        self._callbacks["extra"] = LogParamsAndGrads(
            log_gradient=self.engine.log_extra,
            log_params=self.engine.log_extra,
            log_quantiles=self.engine.log_extra,
            log_every_n_steps=100  # maybe add arg for this?
        )
        self._callbacks["print_epoch"] = PrintEpoch(self.engine.silent)

    def outpath_exclude_keys(self) -> list[str]:
        return [
            self.eval_key,
            "output_dir_name"
        ]

    def _set_outpath(self):
        self._ensure_max_steps()
        base: Path = self.engine.output_dir / self.task.output_dir_name / self.optimizer.output_dir_name
        exclude_keys = self.outpath_exclude_keys()
        exclude_keys += self.engine.outpath_irrelevant_engine_keys()
        diffs = concatenate_dict_keys(dict_differences(self._config, self._default_config), exclude_keys=exclude_keys)
        run_dir = ",".join(f"{k}={str(v)}" for k, v in sorted(diffs.items())) if diffs else "default"
        if len(run_dir) > 254:  # max file name length
            hashdir = hashlib.md5(run_dir.encode()).hexdigest()
            rank_zero_warn(f"folder name {run_dir} is too long, using {hashdir} instead.")
            run_dir = hashdir
        self.run_dir = base / run_dir
        self.checkpoint_dir = self.run_dir / "checkpoints"

    def _generate_configs(self):
        self.engine = EngineConfig(self._config, self.task_key, self.engine_key)
        self.optimizer = OptimizerConfig(self._config, self.optimizer_key, self.task_key, self.identifier_key)
        self.task = TaskConfig(self._config, self.task_key, self.engine_key, self.identifier_key)
        self.evaluation = EvalConfig(
            self._config,
            eval_key=self.eval_key,
            optimizer_key=self.optimizer_key,
            identifier_key=self.identifier_key,
            ignore_keys=self.engine.outpath_irrelevant_engine_keys(prefix=f"{self.engine_key}.") + [f"{self.optimizer_key}.output_dir_name", f"{self.task_key}.output_dir_name"]
        )
