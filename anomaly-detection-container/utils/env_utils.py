# utils/env_utils.py
import os
from pathlib import Path

def in_sagemaker() -> bool:
    return "SM_MODEL_DIR" in os.environ

def data_dir(default: str) -> str:
    # In SageMaker, training data channel 'train' is mounted here
    return os.environ.get("SM_CHANNEL_TRAIN", default)

def model_dir(default: str = "outputs/model") -> str:
    # In SageMaker this is /opt/ml/model (auto-upload to S3 at job end)
    return os.environ.get("SM_MODEL_DIR", default)

def output_dir(default: str = "outputs") -> str:
    return os.environ.get("SM_OUTPUT_DATA_DIR", default)

def ensure_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path
