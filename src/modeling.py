# Import the required libraries.
import json
import copy
import hashlib

import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import LogisticRegression as LGR
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.ensemble import (
    BaggingClassifier as BGC,
    RandomForestClassifier as RFC,
    AdaBoostClassifier as ABC,
    GradientBoostingClassifier as GBC
)

from sklearn.metrics import recall_score
from sklearn.model_selection import RandomizedSearchCV

import utils as utils

import warnings
warnings.filterwarnings("ignore")


# Function to load preprocessed data.
def load_data(data_conf):
    """
    Load every set of data.

    Parameters:
    ----------
    data_conf : list
        Dataset location.

    Returns:
    -------
    X : pd.DataFrame
        Input data features.

    y : pd.Series
        Output data target.
    """

    # Load X and y for each set.
    X = utils.deserialize_data(data_conf[0])
    y = utils.deserialize_data(data_conf[1])

    return X, y

# Function to create training log.
def create_training_log():
    """Return a dictionary representing the training log structure."""
    logger = {
        "model_name": [],
        "model_id": [],
        "training_time": [],
        "training_date": [],
        "train_score": [],
        "cv_score": []
    }

    return logger

# Function to update training log.
def update_training_log(current_log, path_log):
    """
    Update the training log.

    Parameters:
    ----------
    current_log : dict
        The training log current state.

    path_log : str
        The directory of training log.

    Returns:
    -------
    last_log : dict
        The updated training log.
    """

    # Ensure the current log immutable.
    current_log = copy.deepcopy(current_log)

    # Open the training log file.
    try:
        with open(path_log, 'r') as file:
            last_log = json.load(file)
        file.close()
        
    # If the training log does not exists.
    except FileNotFoundError as err:
        # Create the new training log.
        with open(path_log, 'w') as file:
            file.write("[]")
        file.close()

        # Reload the new training log.
        with open(path_log, 'r') as file:
            last_log = json.load(file)
        file.close()

    # Add the current log to previous log.
    last_log.append(current_log)

    # Rewrite the training log with the updated one.
    with open(path_log, 'w') as file:
        json.dump(last_log, file)
        file.close()

    return last_log

# Function to create model object.
def create_model_object():
    """Return a list of model to be fitted."""

    # Create model object.
    knn = KNN()
    lgr = LGR()
    dtc = DTC()
    bgc = BGC()
    rfc = RFC()
    abc = ABC()
    gbc = GBC()

    # Create list of model.
    list_of_model = [
        {"model_name": knn.__class__.__name__, "model_object": knn, "model_id": ""},
        {"model_name": lgr.__class__.__name__, "model_object": lgr, "model_id": ""},
        {"model_name": dtc.__class__.__name__, "model_object": dtc, "model_id": ""},
        {"model_name": bgc.__class__.__name__, "model_object": bgc, "model_id": ""},
        {"model_name": rfc.__class__.__name__, "model_object": rfc, "model_id": ""},
        {"model_name": abc.__class__.__name__, "model_object": abc, "model_id": ""},
        {"model_name": gbc.__class__.__name__, "model_object": gbc, "model_id": ""}
    ]

    return list_of_model

# Function to create hyperparameter space.
def create_param_space():
    """Return a dict of model hyperparameter."""

    # Define each model hyprerparameter space.
    knn_params = {
        "n_neighbors": [2, 3, 4, 5, 6, 10, 15],
        "weights": ["uniform", "distance"],
        "p": [1, 2]
    }

    lgr_params = {
        "C": [0.01, 0.1, 1.0, 10.0]
    }

    # Hyperparameter for DTC, RFC, and GBC.
    DEPTH = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Hyperparameter for BGC, RFC, ABC, and GBC.
    B = [10, 50, 100, 150, 200, 250, 300, 350, 400]
    
    # Hyperparameter for ABC and GBC.
    LR = [0.001, 0.01, 0.05, 0.1, 1]

    dist_params = {
        "KNeighborsClassifier": knn_params,
        "LogisticRegression": lgr_params,
        "DecisionTreeClassifier": {
            "max_depth": DEPTH
        },
        "BaggingClassifier": {
            "n_estimators": B
        },
        "RandomForestClassifier": {
            "n_estimators": B,
            "max_depth": DEPTH
        },
        "AdaBoostClassifier": {
            "n_estimators": B,
            "learning_rate": LR
        },
        "GradientBoostingClassifier": {
            "n_estimators": B,
            "learning_rate": LR,
            "max_depth": DEPTH
        }
    }

    return dist_params

# Function to fit & tune model (CV + HT).
def evaluate_model(models, hyperparameters, X_train, y_train, config, path_log):
    """Cross Validation & Hyperparameter Tuning."""

    # Create training log.
    logger = create_training_log()

    # Define a dictionary to store the trained models.
    trained_models = {}

    # Fit & tune each model.
    for m, h in zip(models, hyperparameters):
        print()
        utils.print_debug(f"Fit & Tune Model : {m['model_name']}...")

        # Create tuner object.
        tuner = RandomizedSearchCV(
            estimator = m["model_object"],
            param_distributions = hyperparameters[h],
            n_iter = 100,
            scoring = "recall",
            cv = 5,
            return_train_score = True,
            n_jobs = -1,
            verbose = 1
        )

        # Compute the training time.
        start_time = utils.time_stamp()
        tuner.fit(X_train, y_train)
        finished_time = utils.time_stamp()

        training_time = finished_time - start_time
        training_time = training_time.total_seconds()

        # Get the model with best hyperparameters.
        best_model = tuner.best_estimator_

        # Get the scores of best model.
        best_index = tuner.best_index_
        train_score = tuner.cv_results_["mean_train_score"][best_index]
        cv_score = tuner.cv_results_["mean_test_score"][best_index]

        # Store the training information.
        logger["model_name"].append(m["model_name"])

        plain_id = str(training_time)
        cipher_id = hashlib.md5(plain_id.encode()).hexdigest()
        logger["model_id"].append(cipher_id)

        logger["training_time"].append(training_time)
        logger["training_date"].append(str(start_time))
        logger["train_score"].append(train_score)
        logger["cv_score"].append(cv_score)

        # Store the best model.
        trained_models[m["model_name"]] = [best_model, train_score, cv_score]

    # Update the current training log.
    training_log = update_training_log(logger, path_log)

    return trained_models, training_log

# Function to show the performance summary.
def training_log_to_df(training_log):
    """Return dataframe of performance summary."""
    performances = pd.DataFrame()

    for log in training_log:
        performances = pd.concat([performances, pd.DataFrame(log)])

    performances = performances.sort_values(
        ["cv_score", "training_time"],
        ascending = [False, True]
    )

    performances = performances.reset_index(drop=True)

    selected_cols = ["model_name", "train_score", "cv_score", "training_time"]
    return performances[selected_cols]


# Main function.
def main():
    # 1. Load configuration file.
    config = utils.load_config()
    utils.print_debug("Config file is loaded...")

    # 2. Load each set of data.
    PATH_DATA_TRAIN = config["path_clean_train"]
    X_train, y_train = load_data(PATH_DATA_TRAIN)
    utils.print_debug("Data train loaded...")

    # 3. Model fit & tune (CV + HT).
    models = create_model_object()
    hyperparameters = create_param_space()

    PATH_TRAINING_LOG = config["path_training_log"]
    trained_models, training_log = evaluate_model(
        models,
        hyperparameters,
        X_train,
        y_train,
        config,
        PATH_TRAINING_LOG
    )
    
    # 4. Get the best model.
    performances = training_log_to_df(training_log)

    best_name = performances["model_name"][0]
    best_model = trained_models[best_name][0]

    utils.print_debug(f"Best Model : {best_name}")

    # 5. Model serialization.
    PATH_PRODUCTION_MODEL = config["path_production_model"]
    utils.serialize_data(best_model, PATH_PRODUCTION_MODEL)


if __name__ == "__main__":
    main()