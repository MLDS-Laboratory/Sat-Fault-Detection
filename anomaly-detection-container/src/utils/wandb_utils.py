# utils/wandb_utils.py
import os
import wandb

def maybe_init_wandb(project: str = "gaf-anomaly-clf", config: dict | None = None):
    """Initialize W&B if WANDB_API_KEY is present; else return a no-op shim."""
    if os.environ.get("WANDB_API_KEY"):
        run = wandb.init(project=os.environ.get("WANDB_PROJECT", project),
                         config=config, reinit=True)
        return run
    class _NoOp:
        def log(self, *a, **k): pass
        def finish(self): pass
        @property
        def config(self): return {}
    return _NoOp()

def log_best_model_as_artifact(filepath: str, name: str = "best-model"):
    if os.environ.get("WANDB_API_KEY"):
        art = wandb.Artifact(name, type="model")
        art.add_file(filepath)
        wandb.log_artifact(art)
