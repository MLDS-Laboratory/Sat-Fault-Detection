from interface_tests import (
    run_generic_satellite_model_tests,
    run_generic_constellation_model_tests,
)

import os
import sys
# === IMPORT MODELS BELOW ===
print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.satellite.outlier_detection import OutlierDetectionModel
from models.constellation.constellation_outlier_detection import ConstellationOutlierDetection

# === FACTORY FUNCTIONS FOR EACH MODEL ===

def sat_outlier_factory():
    return OutlierDetectionModel()

def const_outlier_factory():
    return ConstellationOutlierDetection()

# === RUN TESTS FOR EACH MODEL ===

def test_outlier_model():
    run_generic_satellite_model_tests(sat_outlier_factory)

def test_system_wide_model():
    run_generic_constellation_model_tests(const_outlier_factory)


# Optionally, if you want to be able to run this file standalone:
if __name__ == "__main__":
    test_outlier_model()
    test_system_wide_model()
    print("All tests passed.")
