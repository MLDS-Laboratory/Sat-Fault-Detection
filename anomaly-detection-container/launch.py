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
    p.add_argument("--mode", choices=["local","sagemaker"], default="sagemaker")
    p.add_argument("--data_dir", default="data/ESA-Anomaly/ESA-Mission1")
    p.add_argument("--epochs", type=int, default=10)                # time bottleneck
    p.add_argument("--batch_size", type=int, default=64)            # memory bottleneck
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--model", choices=["pretrained", "scratch"], default="pretrained")
    p.add_argument("--instance_type", default="ml.g5.12xlarge")
    p.add_argument("--spot", action="store_true")
    p.add_argument("--fastfile", action="store_true")
    p.add_argument("--wandb_project", default="gaf-anomaly-clf")
    p.add_argument("--wandb_group", default="baseline")
    p.add_argument("--volume_size", type=int, default=200)  # GB
    p.add_argument("--run_type", choices=["1d", "stacked"], default="1d")
    p.add_argument("--transfer_weights", type=str, default=None, help="S3 or local path to .pth file")
    return p.parse_args()

def run_local(a):
    os.environ["WANDB_PROJECT"] = a.wandb_project
    os.environ["WANDB_RUN_GROUP"] = a.wandb_group
    cmd = (
        f"python src/models/satellite/GAF/gaf_main.py "
        f"--data_dir '{a.data_dir}' --epochs {a.epochs} --batch_size {a.batch_size} --lr {a.lr} --model {a.model}"
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

    # SageMaker Inputs Dictionary
    inputs = {"train": TrainingInput(s3_data=data_input, input_mode=input_mode)}
    
    # If transfer learning, attach the weights as a secondary channel
    if a.transfer_weights:
        weights_input = a.transfer_weights if a.transfer_weights.startswith("s3://") else sess.upload_data(path=a.transfer_weights, key_prefix="gaf-weights")
        inputs["weights"] = TrainingInput(s3_data=weights_input, input_mode="File")

    # Point to the correct entry script based on run_type
    entry = "models/satellite/GAF/stacked_main.py" if a.run_type == "stacked" else "models/satellite/GAF/gaf_main.py"

    estimator = PyTorch(
        entry_point=entry,
        source_dir=code_dir,
        role=role,
        framework_version="2.8",
        py_version="py312",
        instance_type=a.instance_type,
        instance_count=1,
        volume_size=a.volume_size,
        hyperparameters={"epochs": a.epochs, "batch_size": a.batch_size, "lr": a.lr, "model": a.model},
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

python launch.py --mode sagemaker --fastfile --instance_type ml.g5.24xlarge --data_dir s3://mlds-anom-esa/data/ESA-Mission1

python launch.py --mode sagemaker --fastfile --instance_type ml.g5.24xlarge --model scratch--data_dir s3://mlds-anom-esa/data/ESA-Mission1
"""