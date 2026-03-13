# Smoke Detector Service

This repository contains information about Smoke Detector Service using Machine Learning approach.

## Dataset Information

- Dataset downloaded from: [here](https://www.kaggle.com/datasets/deepcontractor/smoke-detection-dataset)
- Dataset description: \
There are around 60.000 data that was collected with the help of *IoT* devices from different scenarios:
    - Normal indoor
    - Normal outdoor
    - Indoor wood fire, firefighter training area
    - Indoor gas fire, firefighter training area
    - Outdoor woord, coal, and gas grill
    - Outdoor high humidity
    - etc.

**Credit to Stefan Blatmann in his project [Real-time Smoke Detection with AI-based Sensor Fusion](https://www.hackster.io/stefanblattmann/real-time-smoke-detection-with-ai-based-sensor-fusion-1086e6)**

- Data Definition:

| **Feature** | **Description** |
|---|---|
| `UTC` (datetime) | Time stamp the sensor records. |
| `Temperature[C]` (float) | Environmemt temperature (celcius). |
| `Humidity[%]` (float) | Percentage of environment temperature. |
| `TVOC[ppb]` (int) | Total Volatile Organic Compounts (parts per billion). |
| `eCO2[ppm]` (int) | CO2 concentration (parts per million). |
| `Raw H2` (int) | Number of H2 gas molecules. |
| `Raw Ethanol` (int) | Number of Ethanol gas molecules. |
| `Pressure[hPa]` (float) | Air pressure (hectopascal, 1hPa = 100Pa). |
| `PM1.0` (float) | Number of particles < 1.0 micrometer. |
| `PM2.5` (float) | Number of particles between 1.0 micrometer and 2.5 micrometer. |
| `NC0.5` (float) | Particle concentrate < 0.5 micrometer. |
| `NC1.0` (float) | Particle concentrate between 0.5 micrometer and 1.0 micrometer. |
| `NC2.5` (float) | Particle concentrate between 1.0 micrometer and 2.5 micrometer. |
| `CNT` (int) | Sample counter. |
---

**Output Variable (to be predicted):**

| **Feature** | **Description** |
|---|---|
| `Fire Alarm` (binary) | (0=no fire detected, 1=fire detected). |

## Predict API (FastAPI)
### Endpoint

`POST` `/predict`

### Description

This API endpoint accepts gas and molecule values as input and returns the predicted fire condition (whether fire exists or not). 

### Request

**Not all variables used in the prediction process**. Below are the example.

```json
{
  "temperature": 35,
  "humidity_pct": 50,
  "pressure": 937,
  "pm10": 10000,
  "tvoc": 50000,
  "co2": 50000,
  "raw_h2": 13000,
  "raw_ethanol": 20000
}
```

### Response

```json
{
  "res": "ADA API!",
  "error_msg": ""
}
```

## Deploy Model in Local
### Cloning the repository
- Use `git clone` to clone this repository, so you can run the code in local environment.
```bash
git clone https://github.com/indraps30/mlp-sd2026
```

### Prerequisites
- Make sure Python and PIP installed in your system (this project use Python 3.12.3 and PIP 26.0.1).
- Create and activate the virtual environment, don't forget to update the package manager.
```bash
python3 -m venv .venv_sd
source .venv_sd/bin/activate
pip install --upgrade pip
```
- Install the required packages from `requirements.txt`.
```bash
pip install -r requirements.txt
```

### Running FastAPI app locally
1. Open a terminal or command prompt.
2. Start the FastAPI app by running the following command:
```bash
python src/api.py
```

This will start the FastAPI server on [http://localhost:8080](http://localhost:8080)

### Running Streamlit app locally
1. Open a new terminal or command prompt.
2. Start the Streamlit app by running the following command:
```bash
streamlit run src/ui.py
```

This will start the Streamlit app on [http://localhost:8501](http://localhost:8501)

Now, both FastAPI server and Streamlit app are running locally. You can interact with the Streamlit app to test and make prediction using the FastAPI and the trained model.

## Retraining Model
You can re-train the model by following these steps:
1. Create new folders `data/interim`, `data/processed`, and `logs`.
2. Make sure to activate the virtual environment and you have installed the required packages.
3. Execute these Python scripts sequentially:
```bash
python src/data_pipeline.py
python src/preprocessing.py
python src/modeling.py
```

Those scripts will re-load the raw dataset, pre-processed it, and re-train the models.

### Docker Build and Run
Or, if you want to use Docker engine, you can simply build a Docker image of this project and get it run with the following commands:
```bash
# make sure you are currently on the project folder.
sudo docker compose up --build
```

PS: tested on WSL2

---
**Last Update: 14-03-2026**