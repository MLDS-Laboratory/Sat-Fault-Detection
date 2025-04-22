import os
import tempfile

def run_generic_satellite_model_tests(model_factory):
    """
    Run generic tests on a SatelliteAnomalyDetectionModel implementation.
    model_factory should be a callable returning an instance of the model.
    """
    # Create a model instance
    model = model_factory()

    # --- Test the detect() method with non-anomalous fake data ---
    detected, details = model.detect(time=100, satellite_id=1, data={"test_metric": 0})
    assert isinstance(detected, bool), "detect() should return a bool as first element"
    assert isinstance(details, list) or details is None, "detect() should return a list or None as second element"

    # --- Test the detect() method with clearly anomalous fake data ---
    # Note: The model’s own logic will decide whether an anomaly is flagged.
    detected, details = model.detect(time=101, satellite_id=1, data={"test_metric": 1})
    detected, details = model.detect(time=102, satellite_id=1, data={"test_metric": 2})
    detected, details = model.detect(time=103, satellite_id=1, data={"test_metric": 0})
    detected, details = model.detect(time=104, satellite_id=1, data={"test_metric": 1})
    detected, details = model.detect(time=105, satellite_id=1, data={"test_metric": 0})
    detected, details = model.detect(time=106, satellite_id=1, data={"test_metric": 999999})
    # If an anomaly is flagged, the details should be a list of objects with the expected attributes.
    if detected:
        for anomaly in details:
            for field in ["satellite_id", "anomaly_model", "time", "time_end", "metric", "value", "message"]:
                assert hasattr(anomaly, field), f"Anomaly detail missing expected field: {field}"

    # --- Test save_model() ---
    # with tempfile.TemporaryDirectory() as tmpdirname:
    #     model_path = os.path.join(tmpdirname, "model_state.txt")
    #     model.save_model(model_path)
    #     assert os.path.exists(model_path), "save_model() should create a file at the given path"

def run_generic_constellation_model_tests(model_factory):
    """
    Run generic tests on a ConstellationAnomalyDetectionModel implementation.
    model_factory should be a callable returning an instance of the model.
    """
    model = model_factory()

    # --- Test the detect() method with non-anomalous fake data ---
    detected, details = model.detect(time=200, satellite_id=2, data={"constellation_metric": 0})
    assert isinstance(detected, bool), "detect() should return a bool as first element"
    assert isinstance(details, list) or details is None, "detect() should return a list or None as second element"

    # --- Test the detect() method with clearly anomalous fake data ---
    detected, details = model.detect(time=101, satellite_id=1, data={"test_metric": 1})
    detected, details = model.detect(time=101, satellite_id=2, data={"test_metric": 1})
    detected, details = model.detect(time=102, satellite_id=1, data={"test_metric": 2})
    detected, details = model.detect(time=102, satellite_id=2, data={"test_metric": 2})
    detected, details = model.detect(time=103, satellite_id=1, data={"test_metric": 0})
    detected, details = model.detect(time=103, satellite_id=2, data={"test_metric": 0})
    detected, details = model.detect(time=104, satellite_id=1, data={"test_metric": 1})
    detected, details = model.detect(time=104, satellite_id=2, data={"test_metric": 1})
    detected, details = model.detect(time=105, satellite_id=1, data={"test_metric": 0})
    detected, details = model.detect(time=105, satellite_id=2, data={"test_metric": 0})
    detected, details = model.detect(time=106, satellite_id=1, data={"test_metric": 2})
    detected, details = model.detect(time=106, satellite_id=2, data={"test_metric": 999999})
    if detected:
        for anomaly in details:
            for field in ["satellite_id", "anomaly_model", "time", "time_end", "metric", "value", "message"]:
                assert hasattr(anomaly, field), f"Anomaly detail missing expected field: {field}"

    # --- Test save_model() ---
    # with tempfile.TemporaryDirectory() as tmpdirname:
    #     model_path = os.path.join(tmpdirname, "constellation_model_state.txt")
    #     model.save_model(model_path)
    #     assert os.path.exists(model_path), "save_model() should create a file at the given path"
