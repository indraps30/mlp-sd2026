# Import the required libraries.
import pandas as pd
import copy

from sklearn.model_selection import train_test_split

import utils as utils


# Function for load raw data.
def load_data(path_data):
    """
    Load csv files and return it as Pandas DataFrame.

    Parameters:
    ----------
    path_data : str
        Raw dataset location.

    Returns:
    -------
    raw_dataset : pd.DataFrame
        Loaded raw dataset.
    """

    # Load csv file.
    raw_dataset = pd.read_csv(path_data)

    # Drop any duplicate data.
    raw_dataset = raw_dataset.drop_duplicates(keep="last")

    return raw_dataset

# Function for data validation.
def data_validation(data):
    """
    Do data validation for removing bad data.

    Parameters:
    ----------
    data : pd.DataFrame
        Loaded raw dataset.

    Returns:
    -------
    data : pd.DataFrame
        Validated data.
    """

    # Ensure raw data immutable.
    data = data.copy()

    # 1. Convert UTC column to datetime type.
    data["UTC"] = pd.to_datetime(data["UTC"], unit="s")

    # 2. Drop CNT column.
    data = data.drop(columns=["CNT"])

    # 3. Rename columns.
    new_names = ["utc", "temperature", "humidity_pct", "tvoc", "co2", "raw_h2", "raw_ethanol", "pressure", "pm10", "pm25", "nc05", "nc10", "nc25", "fire_alarm"]
    data.columns = new_names

    return data

# Function for data defense.
def data_defense(data, config, api=False):
    """
    Do data defense to check data types and range.

    Parameters:
    ----------
    data : pd.DataFrame
        Validated data.

    config : dict
        Loaded configuration file.

    api : bool, default = False
        To check whether the input data from API or not.

    Returns:
    -------
    None, its a void function.
    """

    # Ensure raw data and raw config immutable.
    data = copy.deepcopy(data)
    config = copy.deepcopy(config)

    # Number of data.
    n_data = len(data)

    # List of columns.
    cols_float = config["columns_float"]
    cols_int = config["columns_int"]

    # If the input is not from API.
    if not api:
        # Check data types.
        assert data.select_dtypes("float").columns.to_list() == cols_float, "an error occurs in float columns."
        assert data.select_dtypes("int").columns.to_list() == cols_int, "an error occurs in int columns."

        # Check range of data.
        for col in (cols_float + cols_int):
            min_value = config[f"range_{col}"][0]
            max_value = config[f"range_{col}"][1]
            assert data[col].between(min_value, max_value).sum() == n_data, f"an error occurs in {col} range."

    # In case data defense from API.
    else:
        # Float features used only temperature, humidity_pct, pressure, and pm10.
        del cols_float[4:]

        # Int features used only tvoc, co2, raw_h2, and raw_ethanol.
        del cols_int[4:]

        api_data = data[cols_float + cols_int]

        # Check data types.
        assert api_data.select_dtypes("float").columns.to_list() == cols_float, "an error occurs in float columns."
        assert api_data.select_dtypes("int").columns.to_list() == cols_int, "an error occurs in int columns."

         # Check range of data.
        for col in (cols_float + cols_int):
            min_value = config[f"range_{col}"][0]
            max_value = config[f"range_{col}"][1]
            assert api_data[col].between(min_value, max_value).sum() == n_data, f"an error occurs in {col} range."

# Function for input-output split.
def split_input_output(data, config):
    """
    Split the input (X) and output (y).

    Parameters:
    ----------
    data : pd.DataFrame
        Processed data.

    config : dict
        Loaded configuration parameters.

    Returns:
    -------
    X : pd.DataFrame
        Input data features.

    y : pd.Series
        Output data target.
    """

    # Ensure raw data immutable.
    data = data.copy()

    # Split the X and y.
    X = data[config["features"]]
    y = data[config["label"]]

    return X, y

# Function for train-valid-test split.
def split_train_test(X, y, test_size, random_state=123):
    """
    Split data into train, valid, and test set.

    Parameters:
    ----------
    X : pd.DataFrame
        Input data features.

    y : pd.Series
        Output data target.

    test_size : float
        Proportion of test set.

    random_state : int, default = 123
        For reproducibility.

    Returns:
    -------
    X_train, X_test : pd.DataFrame
        Train and test input.

    y_train, y_test : pd.Series
        Train and test output.
    """

    X_train, X_test, y_train, y_test = train_test_split(
                                            X, y,
                                            test_size = test_size,
                                            random_state = random_state,
                                            stratify = y
                                       )

    return X_train, X_test, y_train, y_test

# Main function.
def main():
    # 1. Load configuration file.
    config = utils.load_config()
    utils.print_debug("Config file is loaded...")

    # 2. Load raw dataset.
    PATH_DATA_RAW = config["path_data_raw"]
    raw_dataset = load_data(PATH_DATA_RAW)
    utils.print_debug("Raw data is loaded...")

    # 3. Data validation.
    validated_data = data_validation(raw_dataset)
    utils.print_debug("Data validation done...")

    # 4. Serialize validated data.
    PATH_DATA_VALIDATED = config["path_data_validated"]
    utils.serialize_data(validated_data, PATH_DATA_VALIDATED)
    utils.print_debug("Validated data is serialized...")

    # 5. Data defense.
    data_defense(validated_data, config)
    utils.print_debug("Data defense done...")

    # 6. Split input-output.
    X, y = split_input_output(validated_data, config)
    utils.print_debug("Split input-output done...")

    # 7. Split train-valid-test.
    RANDOM_STATE = config["random_state"]
    X_train, X_not_train, y_train, y_not_train = split_train_test(
        X, y,
        test_size = 0.2,
        random_state = RANDOM_STATE
    )

    X_valid, X_test, y_valid, y_test = split_train_test(
        X_not_train, y_not_train,
        test_size = 0.5,
        random_state = RANDOM_STATE
    )

    utils.print_debug("Split train-valid-test done...")

    # 8. Data serialization.
    utils.serialize_data(X_train, config["path_data_train"][0])
    utils.serialize_data(y_train, config["path_data_train"][1])
    utils.serialize_data(X_valid, config["path_data_valid"][0])
    utils.serialize_data(y_valid, config["path_data_valid"][1])
    utils.serialize_data(X_test, config["path_data_test"][0])
    utils.serialize_data(y_test, config["path_data_test"][1])
    utils.print_debug("Data pipeline done...")


if __name__ == "__main__":
    main()