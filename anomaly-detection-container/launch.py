import os, argparse
from sagemaker.pytorch import PyTorch
from sagemaker.inputs import TrainingInput
from sagemaker.session import Session
from dotenv import load_dotenv
import logging

logging.getLogger("sagemaker").setLevel(logging.DEBUG)
logging.getLogger("botocore").setLevel(logging.INFO)

load_dotenv()

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["local","sagemaker"], default="local")
    p.add_argument("--data_dir", default="data/ESA-Anomaly/ESA-Mission1")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--instance_type", default="ml.g5.2xlarge")
    p.add_argument("--spot", action="store_true")
    p.add_argument("--fastfile", action="store_true")
    p.add_argument("--wandb_project", default="gaf-anomaly-clf")
    p.add_argument("--wandb_group", default="baseline")
    return p.parse_args()

def run_local(a):
    os.environ["WANDB_PROJECT"] = a.wandb_project
    os.environ["WANDB_RUN_GROUP"] = a.wandb_group
    cmd = (
        f"python src/models/satellite/GAF/gaf_main.py "
        f"--data_dir '{a.data_dir}' --epochs {a.epochs} --batch_size {a.batch_size} --lr {a.lr}"
    )
    raise SystemExit(os.system(cmd))

def run_sagemaker(a):
    sess = Session()
    role = os.environ.get("SAGEMAKER_EXECUTION_ROLE_ARN")
    input_mode = "FastFile" if a.fastfile else "File"

    # If local disk path given, upload to S3:
    data_input = a.data_dir
    if not a.data_dir.startswith("s3://"):
        data_input = sess.upload_data(path=a.data_dir, key_prefix="gaf-data")

    code_dir = os.path.join(os.path.dirname(__file__), "src")

    estimator = PyTorch(
        entry_point="models/satellite/GAF/gaf_main.py",
        source_dir=code_dir,
        role=role,
        framework_version="2.8",
        py_version="py312",
        instance_type=a.instance_type,
        instance_count=1,
        hyperparameters={"epochs": a.epochs, "batch_size": a.batch_size, "lr": a.lr},
        requirements_file="requirements.txt",
        enable_sagemaker_metrics=True,
        use_spot_instances=a.spot,
        max_run=60*60*8,
        max_wait=60*60*12 if a.spot else None,
        environment={
            "WANDB_API_KEY": os.environ.get("WANDB_API_KEY",""),
            "WANDB_PROJECT": a.wandb_project,
            "WANDB_RUN_GROUP": a.wandb_group,
        },
    )
    print("Starting SageMaker training job...")
    estimator.fit(inputs={"train": TrainingInput(s3_data=data_input, input_mode=input_mode)}, logs=["All"])

if __name__ == "__main__":
    args = parse()
    (run_local if args.mode == "local" else run_sagemaker)(args)


"""
python launch.py --mode local --data_dir data/ESA-Anomaly/ESA-Mission1 --epochs 5

python launch.py --mode sagemaker --fastfile --instance_type ml.g5.2xlarge --data_dir s3://mlds-anom-esa/data/ESA-Mission1
"""