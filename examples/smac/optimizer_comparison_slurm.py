from pathlib import Path
import argparse
import json
import time
import subprocess
import sys
from argparse import Namespace
from smac.facade.multi_fidelity_facade import MultiFidelityFacade as SMAC4MF
from smac.intensifier.hyperband import Hyperband
from smac.scenario import Scenario
from ConfigSpace import (
    Configuration,
    ConfigurationSpace,
    Float,
    Integer,
    Constant
)
from pytorch_fob import Engine
from pytorch_fob.engine.utils import set_loglevel, seconds_to_str, str_to_seconds

def job_finshed(job_id: int) -> bool:
    result = subprocess.run(['squeue', '--job', str(job_id)], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, check=False)
    if result.returncode != 0:
        # Job not in queue: probably finished
        return True
    # Parse the output
    output = result.stdout.strip()
    if output:
        lines = output.split('\n')
        if len(lines) > 1:
            # The first line is the header, the second line contains the job status
            status_line = lines[1].split()
            status = status_line[4]  # The fifth column is the job status
            running = status == "PD" or status == "R" or status == "CG" or status == "RS" or status == "RV" or status == "RS"
            finished = status == "CD" or status == "F" or status == "TO" or status == "CA" or status == "NF" or status == "SE" or status == "OOM"
            assert running ^ finished
            return finished
        else:
            # Job not in queue: probably finished
            return True
    else:
        # should output something
        raise Exception("problem with squeue command")


def wait_for_job(job_id: int):
    """Block thread until SLURM job is finished"""
    time.sleep(5)
    while not job_finshed(job_id):
        time.sleep(5)


def config_space(optimizer_name: str) -> ConfigurationSpace:
    cs = ConfigurationSpace()
    cs.add_hyperparameter(Constant("optimizer.name", optimizer_name))
    cs.add_hyperparameter(Float("optimizer.learning_rate", (1.e-5, 1.e-1), log=True))
    cs.add_hyperparameter(Float("optimizer.eta_min_factor", (1.e-3, 1.e-1), log=True))
    cs.add_hyperparameter(Float("optimizer.warmup_factor", (1.e-3, 1.e-0), log=True))
    if optimizer_name in ["adamw_baseline", "sgd_baseline"]:
        cs.add_hyperparameter(Float("optimizer.weight_decay", (1.e-5, 1.e-0), log=True))
    if optimizer_name in ["adamw_baseline", "adamcpr_fast"]:
        cs.add_hyperparameter(Float("optimizer.one_minus_beta1", (1e-2, 2e-1), log=True))
        cs.add_hyperparameter(Float("optimizer.beta2", (0.9, 0.999)))
    if optimizer_name == "sgd_baseline":
        cs.add_hyperparameter(Float("optimizer.momentum", (0, 1)))
    if optimizer_name == "adamcpr_fast":
        cs.add_hyperparameter(Integer("optimizer.kappa_init_param", (1, 19550), log=True))
        cs.add_hyperparameter(Constant("optimizer.kappa_init_method", "warm_start"))
    return cs


def get_target_fn(extra_args, experiment_file):
    def train(config: Configuration, seed: int, budget: float) -> float:
        round_budget = round(budget)
        arglist = extra_args + [f"{k}={v}" for k, v in config.get_dictionary().items()]
        arglist += [
            f"engine.restrict_train_epochs={round_budget}",
            f"engine.seed={seed}",
            "engine.run_scheduler=slurm_jobs",
            "engine.test=false",
            "engine.validate=true",
            "engine.plot=false"
        ]
        engine = Engine()
        engine.parse_experiment_from_file(experiment_file, extra_args=arglist)
        job_ids = engine.run_experiment()
        assert isinstance(job_ids, list) and len(job_ids) == 1
        job_id = job_ids[0]
        run = next(engine.runs())  # only get one run
        wait_for_job(job_id)
        try:
            with open(run.run_dir / "scores.json", "r", encoding="utf8") as f:
                score = json.load(f)
        except FileNotFoundError:
            print("could not load scores, returning inf")
            return float("inf")
        (run.run_dir / "scores.json").unlink()  # delete score so crashed runs later will not yield a score
        result = 1 - sum(map(lambda x: x["val_acc"], score["validation"])) / len(score["validation"])
        print(f"got result: {result}", file=sys.stderr)
        return result
    return train


def run_smac(target_fn, args: Namespace, optimizer_name: str, task_name: str, max_epochs: int, outdir: Path):
    configspace = config_space(optimizer_name)
    n_workers: int = args.n_workers
    scenario = Scenario(
        name=f"FOB_HPO_{task_name}_{optimizer_name}",
        configspace=configspace,
        deterministic=True,
        output_directory=outdir / "smac",
        seed=args.seed,
        n_trials=args.n_trials,
        max_budget=max_epochs,
        min_budget=args.min_budget,
        n_workers=n_workers, # https://github.com/automl/SMAC3/blob/main/examples/1_basics/7_parallelization_cluster.py does not work
    )
    smac = SMAC4MF(
        target_function=target_fn,
        scenario=scenario,
        initial_design=SMAC4MF.get_initial_design(scenario=scenario),
        intensifier=Hyperband(
            scenario=scenario,
            incumbent_selection="highest_budget",
            eta=args.eta,
        ),
        overwrite=True,
    )
    incumbent = smac.optimize()
    return incumbent


def sbatch_time(time: str, time_factor: float) -> str:
    seconds = str_to_seconds(time) if isinstance(time, str) else time
    return seconds_to_str(int(time_factor * seconds))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="runs an experiment specified by a file"
    )
    parser.add_argument("experiment_file", type=Path,
                        help="The yaml file specifying the experiment.")
    parser.add_argument("--log_level", type=str, choices=["debug", "info", "warn", "error", "silent"], default="warn",
                        help="Set the log level")
    parser.add_argument("--n_workers", type=int, default=4,
                        help="maximum number of parallel SMAC runs")
    parser.add_argument("--seed", type=int, default=42,
                        help="seed for SMAC")
    parser.add_argument("--n_trials", type=int, default=200,
                        help="n_trials for SMAC")
    parser.add_argument("--min_budget", type=int, default=5,
                        help="minimum budget for SMAC")
    parser.add_argument("--eta", type=int, default=3,
                        help="eta for Hyperband")
    args, extra_args = parser.parse_known_args()
    set_loglevel(args.log_level)
    experiment_file = args.experiment_file
    engine = Engine()
    engine.parse_experiment_from_file(experiment_file, extra_args=extra_args)
    engine.prepare_data()
    run = next(engine.runs())
    max_epochs = run.task.max_epochs
    optimizer_name = run.optimizer.name
    task_name = run.task.name
    outdir = run.engine.output_dir
    del engine
    incumbent = run_smac(get_target_fn(extra_args, experiment_file), args, optimizer_name, task_name, max_epochs, outdir)
    print(incumbent)
