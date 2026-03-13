# Import the required libraries.
import pandas as pd
import numpy as np

import utils as utils
import preprocessing as preprocessing


# Function to test model prediction from API.
def test_model_prediction_api():
    """Unit test for model prediction from API."""

    # Load serialized estimators.
    config = utils.load_config()
    
    PATH_SCALER = config["path_fitted_scaler"]
    scaler = utils.deserialize_data(PATH_SCALER)

    PATH_PRODUCTION_MODEL = config["path_production_model"]
    best_model = utils.deserialize_data(PATH_PRODUCTION_MODEL)

    # Arrange.
    mock_data = {
        "temperature": [35, 0],
        "humidity_pct": [50, 11],
        "pressure": [937, 931],
        "pm10": [10000, 0],
        "tvoc": [50000, 0],
        "co2": [50000, 400],
        "raw_h2": [13000, 10668],
        "raw_ethanol": [20000, 15317]
    }
    mock_data = pd.DataFrame(mock_data)

    expected_data = np.array([1, 0])

    # Act.
    mock_data = preprocessing.transform_scaler(mock_data, scaler)
    processed_data = best_model.predict(mock_data)

    # Assert.
    assert np.all(processed_data == expected_data)