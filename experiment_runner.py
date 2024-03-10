from pathlib import Path
import argparse
import sys
import time
import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from engine import Engine, Run
from engine.callbacks import LogParamsAndGrads, PrintEpoch
from engine.utils import some, trainer_strategy, begin_timeout, write_results, seconds_to_str


def run_trial(run: Run):
    torch.set_float32_matmul_precision('high')  # TODO: check if gpu has tensor cores
    if not torch.cuda.is_bf16_supported():
        print("Warning: GPU does not support bfloat16, using float16. Results can be different!", file=sys.stderr)
    seed_everything(run.engine.seed, workers=True)
    run.export_config()
    model, data_module = run.get_task()
    model_checkpoint = ModelCheckpoint(
        dirpath=run.run_dir / "checkpoints",
        filename="best-{epoch}-{step}",
        monitor=run.task.target_metric,
        mode=run.task.target_metric_mode,
        save_last=True
    )
    max_epochs = run.task.max_epochs if run.task.max_steps is None else None
    max_steps = some(run.task.max_steps, run.task.max_steps, default=-1)
    devices = some(run.engine.devices, default=run.engine.devices)
    trainer = Trainer(
        max_epochs=max_epochs,
        max_steps=max_steps,
        logger=[
            TensorBoardLogger(
                save_dir=run.run_dir,
                name="tb_logs"
            ),
            CSVLogger(
                save_dir=run.run_dir,
                name="csv_logs"
            )
        ],
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            model_checkpoint,
            LogParamsAndGrads(
                log_gradient=run.engine.log_extra,
                log_params=run.engine.log_extra,
                log_quantiles=run.engine.log_extra,
                log_every_n_steps=100  # maybe add arg for this?
            ),
            PrintEpoch(run.engine.silent)  # TODO: verbosity level
        ],
        devices=devices,
        strategy=trainer_strategy(devices),
        enable_progress_bar=(not run.engine.silent),
        deterministic="warn" if run.engine.deterministic else False,
        precision="bf16-mixed" if torch.cuda.is_bf16_supported() else "16-mixed"
    )
    tester = Trainer(
        devices=1,
        enable_progress_bar=(not run.engine.silent),
        deterministic="warn" if run.engine.deterministic else False,
        precision="bf16-mixed" if torch.cuda.is_bf16_supported() else "16-mixed"
    )
    if run.engine.test_only:
        ckpt_path = run.engine.resume
        mode = "final" if ckpt_path is None or ckpt_path.stem.startswith("last") else "best"
        score = tester.test(model, datamodule=data_module, ckpt_path=ckpt_path)
        write_results(score, run.engine.output_dir / f"results_{mode}_model.json")
    else:
        start_time = time.time()
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True,
            enable_math=True,
            enable_mem_efficient=(run.engine.optimize_memory or not run.engine.deterministic)
        ):
            trainer.fit(model, datamodule=data_module, ckpt_path=run.engine.resume)
        end_time = time.time()
        train_time = int(end_time - start_time)
        print(f"Finished training in {seconds_to_str(train_time)}.")
        final_score = tester.test(model, datamodule=data_module)
        best_score = tester.test(model, datamodule=data_module, ckpt_path=model_checkpoint.best_model_path)
        write_results(final_score, run.run_dir / "results_final_model.json")
        write_results(best_score, run.run_dir / "results_best_model.json")


def main(args: argparse.Namespace, extra_args: list[str]):
    engine = Engine()
    engine.parse_experiment(args.experiment_file, extra_args=extra_args)
    runs = engine.runs()
    for i, run in enumerate(runs):
        print(f"Starting run {i + 1}/{len(runs)}.")
        run_trial(run)

    if args.send_timeout:
        print("submission_runner.py finished! Setting timeout of 10 seconds, as tqdm sometimes is stuck\n")
        begin_timeout()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="runs an experiment specified by a file"
    )
    parser.add_argument("experiment_file", type=Path,
                        help="The yaml file specifying the experiment.")
    parser.add_argument("--send_timeout", action="store_true",
                        help="send a timeout after finishing this script (if you have problems with tqdm being stuck)")
    args, extra_args = parser.parse_known_args()
    main(args, extra_args)
