import json
import os
import shutil
import time

EXPERIMENTS_ROOT = "experiments"


def resolve_checkpoint(checkpoint_file: str, experiment_name: str = None) -> str:
    """
    Returns the path of a checkpoint. The path is used as given if it exists,
    otherwise it is looked up inside the directory of the experiment.
    """
    if os.path.exists(checkpoint_file):
        return checkpoint_file
    if experiment_name:
        candidate = os.path.join(EXPERIMENTS_ROOT, experiment_name, checkpoint_file)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"Checkpoint '{checkpoint_file}' not found (experiment: {experiment_name})")


def create_experiment_dir(config_path: str, model: str, experiment_name: str = None, resume: bool = False) -> str:
    """
    Creates (or reuses, when resuming) the directory of an experiment:
        experiments/<name>/config.yaml     copy of the configuration file
        experiments/<name>/best.pth        best checkpoint
        experiments/<name>/last.pth        last checkpoint
        experiments/<name>/metrics.jsonl   one json line per epoch

    The name is `experiment_name` if given, otherwise `<model>_<timestamp>`.
    An existing directory is only reused when resuming a training.
    """
    stamp = time.strftime("%Y%m%d_%H%M%S")
    name = experiment_name or f"{model}_{stamp}"
    path = os.path.join(EXPERIMENTS_ROOT, name)

    if os.path.exists(path):
        if not resume:
            raise FileExistsError(f"Experiment '{path}' already exists. Choose another experiment_name, or set from_checkpoint to true to resume it.")
        config_copy = os.path.join(path, f"config_{stamp}.yaml")    # keeps the original configuration
    else:
        os.makedirs(path)
        config_copy = os.path.join(path, "config.yaml")

    shutil.copyfile(config_path, config_copy)
    return path


def log_metrics(experiment_dir: str, metrics: dict):
    with open(os.path.join(experiment_dir, "metrics.jsonl"), "a") as f:
        f.write(json.dumps(metrics) + "\n")
