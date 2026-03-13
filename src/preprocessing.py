# Import the required libraries.
import pandas as pd

from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

import utils as utils
import data_pipeline as data_pipeline


# Function for load data.
def load_data(data_conf):
    """
    Load every set of data.

    Parameters:
    ----------
    data_conf : list
        Dataset location.

    Returns:
    -------
    data : pd.DataFrame
        Loaded dataset.
    """

    # Load X and y for each set.
    X = utils.deserialize_data(data_conf[0])
    y = utils.deserialize_data(data_conf[1])

    # Concatenate X and y for each set.
    data = pd.concat([X, y], axis=1)

    return data

# Function for outliers removal.
def remove_outliers(data):
    """
    IQR-based outliers removal.

    Parameters:
    ----------
    data : pd.DataFrame
        Loaded data.

    Returns:
    -------
    data_clean : pd.DataFrame
        Cleaned data.
    """

    # Ensure raw data immutable.
    data = data.copy()
    list_cleaned_data = []

    # Calculate IQR for each feature (exclude label).
    for col in data.columns[:-1]:
        q1 = data[col].quantile(0.25)
        q3 = data[col].quantile(0.75)
        iqr = q3 - q1

        # Boolean condition for outliers.
        q1_cond = data[col] < (q1 - (1.5 * iqr))
        q3_cond = data[col] > (q3 + (1.5 * iqr))

        clean = data[~(q1_cond | q3_cond)]
        list_cleaned_data.append(clean)

    data_clean = pd.concat(list_cleaned_data)
    n_duplicated_idx = data_clean.index.value_counts()
    used_idx = n_duplicated_idx[n_duplicated_idx == (data.shape[1]-1)].index
    data_clean = data_clean.loc[used_idx].drop_duplicates()

    return data_clean

# Function to fit the scaler.
def fit_scaler(data, path_scaler):
    """
    Fit the scaler.
    
    Parameters:
    ----------
    data : pd.DataFrame
        Input data (all features must be in numeric form)

    path_scaler : str
        The scaler location.
        
    Returns:
    -------
    scaler : sklearn.preprocessing.StandardScaler
        Fitted scaler object (storing the mean & std of all features)
    """

    # Create scaler object.
    scaler = StandardScaler()

    # Fit the scaler.
    scaler.fit(data)

    # Serialize the scaler.    
    utils.serialize_data(scaler, path_scaler)
    
    return scaler

# Function to scale the data.
def transform_scaler(data, scaler):
    """
    Transform the data using scaler.
    
    Parameters:
    ----------
    data : pd.DataFrame
        Input data (all features must be in numeric form)    
        
    scaler : sklearn.preprocessing.StandardScaler
        Fitted scaler object (storing the mean & std of all features)
        
    Returns:
    -------
    data : pd.DataFrame
        The scaled data
    """

    # Ensure raw data immutable.
    data = data.copy()

    # Scale the data.
    scaled_data = scaler.transform(data)

    # Convert to dataframe.
    X_scaled = pd.DataFrame(
        scaled_data,
        columns = data.columns,
        index = data.index
    )
    
    return X_scaled

# Function for label balancing. 
def label_balancer(X, y, config, random_state=123):
    """
    Balancing the category label.

    Parameters:
    ----------
    X : pd.DataFrame
        The scaled data.

    y : pd.DataFrame
        The label to be balanced.

    config : dict
        The loaded configuration file.

    random_state : int, default = 123
        For reproducibility.

    Returns:
    -------
    X_balanced : pd.DataFrame
        The features with balanced label.

    y_balanced : pd.Series
        The label with balanced label.
    """

    # Ensure raw data immutable.
    X = X.copy()
    y = y.copy()
    label = config["label"]

    # Set the balancer.
    balancer = SMOTE(random_state = random_state)

    # Fit resample the balancer.
    X_balanced, y_balanced = balancer.fit_resample(X, y)

    # Convert to pandas format.
    X_balanced = pd.DataFrame(X_balanced, columns=X.columns)
    y_balanced = pd.Series(y_balanced, name=label)

    return X_balanced, y_balanced


# Main function.
def main():
    # 1. Load configuration file.
    config = utils.load_config()
    utils.print_debug("Config file is loaded...")

    # 2. Load each set of data.
    PATH_DATA_TRAIN = config["path_data_train"]
    data_train = load_data(PATH_DATA_TRAIN)
    utils.print_debug("Data train loaded...")

    PATH_DATA_VALID = config["path_data_valid"]
    data_valid = load_data(PATH_DATA_VALID)
    utils.print_debug("Data valid loaded...")

    PATH_DATA_TEST = config["path_data_test"]
    data_test = load_data(PATH_DATA_TEST)
    utils.print_debug("Data test loaded...")

    # 3. Outliers removal.
    utils.print_debug("PREPROCESSING - START")
    data_train_clean = remove_outliers(data_train)
    utils.print_debug("Outliers removal done...")

    # 4. Split input-output.
    X_train_clean, y_train_clean = data_pipeline.split_input_output(data_train_clean, config)
    X_valid, y_valid = data_pipeline.split_input_output(data_valid, config)
    X_test, y_test = data_pipeline.split_input_output(data_test, config)
    utils.print_debug("Split input-output done...")

    # 5. Scale data.
    PATH_SCALER = config["path_fitted_scaler"]
    scaler = fit_scaler(X_train_clean, PATH_SCALER)

    X_train_scaled = transform_scaler(X_train_clean, scaler)
    X_valid_scaled = transform_scaler(X_valid, scaler)
    X_test_scaled = transform_scaler(X_test, scaler)
    utils.print_debug("Scaling data done...")

    # 6. Label balancing.
    RANDOM_STATE = config["random_state"]
    X_sm, y_sm = label_balancer(
        X_train_scaled, y_train_clean,
        config = config,
        random_state = RANDOM_STATE
    )
    utils.print_debug("Label balancing done...")

    # 7. Data serialization.
    utils.serialize_data(X_sm, config["path_clean_train"][0])
    utils.serialize_data(y_sm, config["path_clean_train"][1])
    utils.serialize_data(X_valid, config["path_clean_valid"][0])
    utils.serialize_data(y_valid, config["path_clean_valid"][1])
    utils.serialize_data(X_test, config["path_clean_test"][0])
    utils.serialize_data(y_test, config["path_clean_test"][1])
    utils.print_debug("PREPROCESSING - END")


if __name__ == "__main__":
    main()