import kagglehub
import os

current_path = os.path.dirname(os.path.abspath(__file__))

# check if the nasa-smap-msl directory exists
if not os.path.exists(os.path.join(current_path, "nasa-smap-msl")):
    os.makedirs(os.path.join(current_path, "nasa-smap-msl"))

# get absolute path to the nasa-smap-msl directory
nasa_smap_msl_path = os.path.join(current_path, "nasa-smap-msl")

path = kagglehub.dataset_download("patrickfleith/nasa-anomaly-detection-dataset-smap-msl", path=nasa_smap_msl_path)

print("Path to NASA SMAP/MSL dataset files:", path)