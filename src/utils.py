# Import the required libraries.
import yaml
import joblib

from datetime import datetime

# Constant variables.
PATH_CONFIG = "./config/config.yaml"


# Common functions.
# Function to load configuration file.
def load_config():
    """Load the configuration file (config.yaml)."""

    # Try to load config.yaml file.
    try:
        with open(PATH_CONFIG, 'r') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError as err:
        raise RuntimeError(f"Configuration file not found in {PATH_CONFIG}")

    return config

# Function to update configuration file.
def update_config(key, value, config):
    """
    Update the configuration parameter values.

    Parameters:
    ----------
    key : str
        Key to be updated.

    value : any type supported in Python
        Updated value.

    config : dict
        Loaded configuration file.

    Returns:
    -------
    config : dict
        Updated configuration file.
    """

    # Ensure raw config file immutable.
    config = config.copy()

    # Update configuration parameters.
    config[key] = value

    # Rewrite configuration file.
    with open(PATH_CONFIG, 'w') as file:
        yaml.dump(config, file)

    print(f"Params Updated! \nKey: {key} \nValue: {value}\n")

    # Reload updated configuration file.
    config = load_config()

    return config

# Function to serialize data.
def serialize_data(data, path):
    """
    Dump data into pickle file.

    Parameters:
    data : pd.DataFrame or sklearn object
        Data to be serialize.

    path : str
        Serialized data location.

    Returns:
    -------
    None, its a void function.
    """

    print(f"Data serialized to {path}")
    return joblib.dump(data, path)

# Function to deserialize data.
def deserialize_data(path):
    """
    Load and return pickle file.

    Parameters:
    ----------
    path : str
        Serialized data location.

    Returns:
    -------
    pd.DataFrame or sklearn object.
    """

    print(f"Data deserialized from {path}")
    return joblib.load(path)

# Function to show current datetime.
def time_stamp():
    """Return current datetime."""
    return datetime.now()
    

# Function for debugging in terminal.
def print_debug(message):
    """
    Print time stamp with message on terminal.

    Parameters:
    ----------
    message : str
        Text to be printed.

    Returns:
    -------
    None, its a void function.
    """

    print(time_stamp(), message)