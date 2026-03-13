# Import the required libraries.
from fastapi import FastAPI
from pydantic import BaseModel

import uvicorn
import pandas as pd

import utils as utils
import data_pipeline as data_pipeline
import preprocessing as preprocessing


# Load serialized estimators.
config = utils.load_config()
scaler = utils.deserialize_data(config["path_fitted_scaler"])
best_model = utils.deserialize_data(config["path_production_model"])


# Define input data structure.
class DataAPI(BaseModel):
    """Represents the user input data structure."""
    temperature : float
    humidity_pct : float
    pressure : float
    pm10 : float
    tvoc : int
    co2 : int
    raw_h2 : int
    raw_ethanol : int

# Create API object.
app = FastAPI()

# Define handlers.
@app.get("/")
def home():
    return {"message": "Hello, FastAPI up!"}

@app.post("/predict/")
def predict(data: DataAPI):
    # Convert DataAPI to Pandas DataFrame.
    data = pd.DataFrame(data).set_index(0).T.reset_index(drop=True)

    # Convert dtype.
    data = pd.concat(
        [
            data[config["features"][:4]].astype(float),
            data[config["features"][4:]].astype(int)
        ],
        axis = 1
    )

    # Perform data defense.
    try:
        data_pipeline.data_defense(data, config, api=True)
    except AssertionError as err:
        return {"res": [], "error_msg": str(err)}

    # Scale the data.
    data = preprocessing.transform_scaler(data, scaler)

    # Predict data.
    y_pred = best_model.predict(data)

    if y_pred[0] == 0:
        y_pred = "TIDAK ADA API"
    else:
        y_pred = "ADA API!"
    
    return {"res": y_pred, "error_msg": ""}


if __name__ == "__main__":
    uvicorn.run("api:app", host = "0.0.0.0", port = 8080)