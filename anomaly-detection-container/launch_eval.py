import os
import argparse
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
    p.add_argument("--mode", choices=["local", "sagemaker"], default="sagemaker")
    p.add_argument("--data_dir", required=True, help="S3 or local path to the target dataset (e.g., Mission 2)")
    p.add_argument("--weights", required=True, help="S3 or local path to the .pth file trained on Mission 1")
    p.add_argument("--model", choices=["pretrained", "scratch"], default="pretrained")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--instance_type", default="ml.g5.8xlarge") 
    p.add_argument("--spot", action="store_true")
    p.add_argument("--fastfile", action="store_true")
    p.add_argument("--wandb_project", default="gaf-anomaly-clf")
    p.add_argument("--wandb_group", default="zero-shot-eval")
    p.add_argument("--volume_size", type=int, default=128)  # GB
    return p.parse_args()

def run_local(a):
    os.environ["WANDB_PROJECT"] = a.wandb_project
    os.environ["WANDB_RUN_GROUP"] = a.wandb_group
    os.environ["SM_CHANNEL_DATASET"] = a.data_dir
    
    # Locally, we can just map the weights dir to the folder containing the file
    weights_dir = os.path.dirname(os.path.abspath(a.weights))
    os.environ["SM_CHANNEL_WEIGHTS"] = weights_dir
    
    cmd = (
        f"python src/models/satellite/GAF/gaf_eval.py "
        f"--batch_size {a.batch_size} --model {a.model}"
    )
    raise SystemExit(os.system(cmd))

def run_sagemaker(a):
    sess = Session()
    role = os.environ.get("SAGEMAKER_EXECUTION_ROLE_ARN")
    input_mode = "FastFile" if a.fastfile else "File"

    # 1. Handle Dataset Upload
    data_input = a.data_dir
    if not a.data_dir.startswith("s3://"):
        data_input = sess.upload_data(path=a.data_dir, key_prefix="gaf-eval-data")

    # 2. Handle Weights Upload
    weights_input = a.weights
    if not a.weights.startswith("s3://"):
        weights_input = sess.upload_data(path=a.weights, key_prefix="gaf-eval-weights")

    code_dir = os.path.join(os.path.dirname(__file__), "src")

    estimator = PyTorch(
        entry_point="models/satellite/GAF/gaf_eval.py",
        source_dir=code_dir,
        role=role,
        framework_version="2.8",
        py_version="py312",
        instance_type=a.instance_type,
        instance_count=1,
        hyperparameters={"batch_size": a.batch_size, "model": a.model},
        requirements_file="requirements.txt",
        use_spot_instances=a.spot,
        max_run=60*60*4, # Eval should be fast, kill it if it hangs
        volume_size=a.volume_size,
        environment={
            "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", ""),
            "WANDB_PROJECT": a.wandb_project,
            "WANDB_RUN_GROUP": a.wandb_group,
        },
    )
    
    print("Starting SageMaker Evaluation Job...")
    
    # Pass both channels. SageMaker maps them to SM_CHANNEL_DATASET and SM_CHANNEL_WEIGHTS
    estimator.fit(
        inputs={
            "dataset": TrainingInput(s3_data=data_input, input_mode=input_mode),
            "weights": TrainingInput(s3_data=weights_input, input_mode="File")
        }, 
        logs=["All"]
    )

if __name__ == "__main__":
    args = parse()
    (run_local if args.mode == "local" else run_sagemaker)(args)


"""

python launch_eval.py --mode sagemaker --fastfile --data_dir s3://mlds-anom-esa/data/ESA-Mission2 --weights s3://mlds-anom-esa/weights/Mission1_PretrainedResNet_best.pth --model pretrained

"""