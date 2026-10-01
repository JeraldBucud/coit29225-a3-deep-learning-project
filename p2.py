"""
COIT29225 Neural Networks & Deep Learning
Assessment 3 - Task 2.2
Australian Stock Price Forecasting Using LSTM

This program develops a Long Short-Term Memory (LSTM) neural network
for forecasting Australian stock closing prices.

The student's ID ends in an odd digit, therefore the assigned mining
group is used:

- RIO - Rio Tinto Group
- BHP - BHP Group
- FMG - Fortescue Ltd.

The supplied CSV files contain 254 trading days and are stored in
reverse chronological order, from the newest date to the oldest date.

The program currently performs the following stages:

1. Loads the assigned stock CSV files.
2. Extracts the Date and Close $ columns.
3. Validates dates and closing prices.
4. Converts the data into chronological order.
5. Uses the first 204 observations as the historical training period.
6. Uses the remaining 50 observations as the forecasting/testing period.
7. Scales closing prices using historical training data only.
8. Generates LSTM sequences for time steps 5 and 10.
9. Creates chronological training and validation subsets.
10. Builds and compiles LSTM regression models.
11. Supports a short LSTM smoke test.
12. Runs six controlled neuron/epoch configurations.
13. Evaluates all configurations using validation RMSE in stock-price dollars.
14. Identifies the best configuration for each stock and time step.
15. Retrains each selected model using all 204 historical observations.
16. Recursively forecasts the final 50 trading days.
17. Calculates final forecast RMSE in Australian dollars.
18. Saves final prediction tables, loss curves, forecast plots, and models.

The assessment-required interactive menu is added after the final
forecasting workflow has been validated.
"""

from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------

# Directory containing the supplied Assessment 3 stock-price CSV files.
DATA_DIR = Path("data")

# Directory used for generated experiment outputs.
OUTPUT_DIR = Path("outputs")

# The student's ID ends in an odd digit, so the mining group is required.
STOCK_FILES = {
    "RIO": DATA_DIR / "RIO.csv",
    "BHP": DATA_DIR / "BHP.csv",
    "FMG": DATA_DIR / "FMG.csv",
}

# Human-readable stock names used in terminal output and report figures.
STOCK_NAMES = {
    "RIO": "Rio Tinto Group",
    "BHP": "BHP Group",
    "FMG": "Fortescue Ltd.",
}

# Columns required by the assessment.
DATE_COLUMN = "Date"
CLOSE_COLUMN = "Close $"

# The supplied dataset contains one trading year.
EXPECTED_RECORDS = 254

# Assessment-required chronological split.
TRAINING_DAYS = 204
FORECAST_DAYS = 50

# Assessment-required LSTM time-step settings.
TIME_STEPS = (
    5,
    10,
)

# Fraction of generated historical sequences reserved for
# chronological validation during parameter tuning.
VALIDATION_FRACTION = 0.20

# Fixed random seed used to improve experiment reproducibility.
RANDOM_SEED = 42

# Development-only LSTM smoke-test settings.
SMOKE_TEST_LSTM_UNITS = 32
SMOKE_TEST_EPOCHS = 2
SMOKE_TEST_BATCH_SIZE = 16

# Default optimiser learning rate.
LEARNING_RATE = 0.001


# ---------------------------------------------------------------------------
# LSTM Parameter Experiment Configurations
# ---------------------------------------------------------------------------

# Six controlled configurations are used to investigate the effect of:
#
# - LSTM neuron count
# - training epoch count
#
# Batch size and learning rate remain fixed so the effect of changing
# neurons and epochs can be compared more clearly.
LSTM_CONFIGURATIONS = [
    {
        "name": "A_16_units_20_epochs",
        "lstm_units": 16,
        "epochs": 20,
        "batch_size": 16,
        "learning_rate": 0.001,
    },
    {
        "name": "B_16_units_50_epochs",
        "lstm_units": 16,
        "epochs": 50,
        "batch_size": 16,
        "learning_rate": 0.001,
    },
    {
        "name": "C_32_units_20_epochs",
        "lstm_units": 32,
        "epochs": 20,
        "batch_size": 16,
        "learning_rate": 0.001,
    },
    {
        "name": "D_32_units_50_epochs",
        "lstm_units": 32,
        "epochs": 50,
        "batch_size": 16,
        "learning_rate": 0.001,
    },
    {
        "name": "E_64_units_20_epochs",
        "lstm_units": 64,
        "epochs": 20,
        "batch_size": 16,
        "learning_rate": 0.001,
    },
    {
        "name": "F_64_units_50_epochs",
        "lstm_units": 64,
        "epochs": 50,
        "batch_size": 16,
        "learning_rate": 0.001,
    },
]



# ---------------------------------------------------------------------------
# Selected LSTM Configurations
# ---------------------------------------------------------------------------

# These configurations produced the lowest chronological validation RMSE
# during the completed 36-run parameter experiment.
SELECTED_LSTM_CONFIGURATIONS = {
    ("RIO", 5): "C_32_units_20_epochs",
    ("RIO", 10): "A_16_units_20_epochs",
    ("BHP", 5): "F_64_units_50_epochs",
    ("BHP", 10): "D_32_units_50_epochs",
    ("FMG", 5): "F_64_units_50_epochs",
    ("FMG", 10): "F_64_units_50_epochs",
}


# ---------------------------------------------------------------------------
# 2. Stock Data Loading and Validation
# ---------------------------------------------------------------------------

def load_stock_data(
    stock_code,
    file_path,
):
    """
    Load and validate one stock-price CSV file.

    Only the Date and Close $ columns are retained because the assessment
    uses historical closing prices for LSTM forecasting.

    Dates are parsed using day-first formatting because the supplied data
    uses Australian-style dates such as 16/09/2026.

    Parameters
    ----------
    stock_code : str
        Stock ticker such as RIO, BHP, or FMG.

    file_path : pathlib.Path
        Path to the stock CSV file.

    Returns
    -------
    pandas.DataFrame
        Validated stock data ordered chronologically from oldest to newest.

    Raises
    ------
    FileNotFoundError
        If the required CSV file does not exist.

    ValueError
        If required columns are missing or invalid records are detected.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Stock data file not found: {file_path}"
        )

    # RIO and FMG contain additional market columns while BHP contains
    # only Date and Close $. Load first, then retain common required data.
    raw_data = pd.read_csv(
        file_path
    )

    required_columns = {
        DATE_COLUMN,
        CLOSE_COLUMN,
    }

    missing_columns = (
        required_columns
        - set(raw_data.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{stock_code}: missing required columns: "
            f"{sorted(missing_columns)}"
        )

    stock_data = raw_data[
        [
            DATE_COLUMN,
            CLOSE_COLUMN,
        ]
    ].copy()

    # Parse Australian-style dates.
    stock_data[DATE_COLUMN] = pd.to_datetime(
        stock_data[DATE_COLUMN],
        dayfirst=True,
        errors="coerce",
    )

    # Ensure closing prices are numeric.
    stock_data[CLOSE_COLUMN] = pd.to_numeric(
        stock_data[CLOSE_COLUMN],
        errors="coerce",
    )

    invalid_dates = (
        stock_data[DATE_COLUMN]
        .isna()
        .sum()
    )

    invalid_prices = (
        stock_data[CLOSE_COLUMN]
        .isna()
        .sum()
    )

    if invalid_dates > 0:
        raise ValueError(
            f"{stock_code}: "
            f"{invalid_dates} invalid date value(s) detected."
        )

    if invalid_prices > 0:
        raise ValueError(
            f"{stock_code}: "
            f"{invalid_prices} invalid closing price value(s) detected."
        )

    duplicate_dates = (
        stock_data[DATE_COLUMN]
        .duplicated()
        .sum()
    )

    if duplicate_dates > 0:
        raise ValueError(
            f"{stock_code}: "
            f"{duplicate_dates} duplicate trading date(s) detected."
        )

    # The supplied files run from newest to oldest.
    # Sorting ascending converts them to proper chronological order.
    stock_data = stock_data.sort_values(
        by=DATE_COLUMN,
        ascending=True,
    ).reset_index(
        drop=True
    )

    return stock_data


# ---------------------------------------------------------------------------
# 3. Training and Forecast Split
# ---------------------------------------------------------------------------

def split_stock_data(
    stock_data,
):
    """
    Divide one chronological stock series into training and forecast sets.

    The first 204 trading days are used as historical training data.
    The final 50 trading days are reserved as future forecasting/testing
    data and must remain untouched during model tuning.

    Parameters
    ----------
    stock_data : pandas.DataFrame
        Chronologically ordered stock-price data.

    Returns
    -------
    tuple
        Training DataFrame and forecasting/testing DataFrame.
    """

    required_records = (
        TRAINING_DAYS
        + FORECAST_DAYS
    )

    if len(stock_data) < required_records:
        raise ValueError(
            "Insufficient stock data. "
            f"Required at least {required_records} records, "
            f"but found {len(stock_data)}."
        )

    training_data = stock_data.iloc[
        :TRAINING_DAYS
    ].copy()

    forecast_data = stock_data.iloc[
        TRAINING_DAYS:
        TRAINING_DAYS + FORECAST_DAYS
    ].copy()

    return (
        training_data,
        forecast_data,
    )


# ---------------------------------------------------------------------------
# 4. Closing-Price Scaling
# ---------------------------------------------------------------------------

def prepare_scaled_prices(
    training_data,
    forecast_data,
):
    """
    Scale stock closing prices using historical training data only.

    MinMaxScaler is fitted exclusively on the first 204 observations.
    The same fitted scaler is then applied to the 50 future observations.

    This prevents future test information from leaking into preprocessing.

    Parameters
    ----------
    training_data : pandas.DataFrame
        First 204 chronological observations.

    forecast_data : pandas.DataFrame
        Final 50 observations reserved for forecasting/testing.

    Returns
    -------
    tuple
        Fitted scaler, scaled training prices, and scaled forecast prices.
    """

    training_prices = training_data[
        [CLOSE_COLUMN]
    ].to_numpy(
        dtype=np.float32
    )

    forecast_prices = forecast_data[
        [CLOSE_COLUMN]
    ].to_numpy(
        dtype=np.float32
    )

    scaler = MinMaxScaler(
        feature_range=(0, 1)
    )

    # Fit only on the historical training period.
    scaled_training_prices = scaler.fit_transform(
        training_prices
    )

    # Transform future prices without refitting.
    #
    # Future values may be below 0 or above 1 if prices move outside
    # the range observed during the first 204 days.
    scaled_forecast_prices = scaler.transform(
        forecast_prices
    )

    return (
        scaler,
        scaled_training_prices,
        scaled_forecast_prices,
    )


# ---------------------------------------------------------------------------
# 5. LSTM Sequence Creation
# ---------------------------------------------------------------------------

def create_lstm_sequences(
    scaled_prices,
    time_step,
):
    """
    Convert a scaled stock-price series into supervised LSTM sequences.

    Example with time step 5:

        [day 1, day 2, day 3, day 4, day 5] -> day 6
        [day 2, day 3, day 4, day 5, day 6] -> day 7

    Parameters
    ----------
    scaled_prices : numpy.ndarray
        Scaled closing prices with shape (observations, 1).

    time_step : int
        Number of previous trading days used to predict the next day.

    Returns
    -------
    tuple
        LSTM input tensor X and target vector y.
    """

    if time_step <= 0:
        raise ValueError(
            "Time step must be greater than zero."
        )

    if len(scaled_prices) <= time_step:
        raise ValueError(
            "Insufficient observations for the requested "
            f"time step of {time_step}."
        )

    sequences = []
    targets = []

    for index in range(
        time_step,
        len(scaled_prices),
    ):
        sequence = scaled_prices[
            index - time_step:index
        ]

        target = scaled_prices[
            index,
            0,
        ]

        sequences.append(
            sequence
        )

        targets.append(
            target
        )

    X = np.asarray(
        sequences,
        dtype=np.float32,
    )

    y = np.asarray(
        targets,
        dtype=np.float32,
    )

    return X, y


def prepare_lstm_data(
    training_data,
    forecast_data,
    time_step,
):
    """
    Prepare scaled stock prices and historical LSTM sequences.

    Parameters
    ----------
    training_data : pandas.DataFrame
        First 204 historical observations.

    forecast_data : pandas.DataFrame
        Final 50 future observations.

    time_step : int
        Number of previous trading days supplied to the LSTM.

    Returns
    -------
    dict
        Scaler, scaled datasets, and generated historical sequences.
    """

    (
        scaler,
        scaled_training_prices,
        scaled_forecast_prices,
    ) = prepare_scaled_prices(
        training_data,
        forecast_data,
    )

    X_sequences, y_sequences = create_lstm_sequences(
        scaled_training_prices,
        time_step,
    )

    return {
        "scaler": scaler,
        "scaled_training": scaled_training_prices,
        "scaled_forecast": scaled_forecast_prices,
        "X_train": X_sequences,
        "y_train": y_sequences,
    }


# ---------------------------------------------------------------------------
# 6. Chronological Training and Validation Split
# ---------------------------------------------------------------------------

def split_lstm_sequences(
    X,
    y,
    validation_fraction=VALIDATION_FRACTION,
):
    """
    Split historical sequences chronologically into train/validation sets.

    Earlier sequences are used for model fitting and later sequences are
    reserved for validation.

    Random splitting is deliberately avoided because these are time-series
    observations.

    Parameters
    ----------
    X : numpy.ndarray
        Complete historical LSTM sequence tensor.

    y : numpy.ndarray
        Corresponding next-day targets.

    validation_fraction : float, optional
        Fraction of sequences reserved for validation.

    Returns
    -------
    tuple
        X_train, X_validation, y_train, y_validation.
    """

    if not 0 < validation_fraction < 1:
        raise ValueError(
            "Validation fraction must be between 0 and 1."
        )

    split_index = int(
        len(X)
        * (1 - validation_fraction)
    )

    if split_index <= 0:
        raise ValueError(
            "Training split contains no sequences."
        )

    if split_index >= len(X):
        raise ValueError(
            "Validation split contains no sequences."
        )

    X_train = X[
        :split_index
    ]

    X_validation = X[
        split_index:
    ]

    y_train = y[
        :split_index
    ]

    y_validation = y[
        split_index:
    ]

    return (
        X_train,
        X_validation,
        y_train,
        y_validation,
    )


# ---------------------------------------------------------------------------
# 7. LSTM Model Architecture
# ---------------------------------------------------------------------------

def build_lstm_model(
    time_step,
    lstm_units,
):
    """
    Build a simple LSTM regression model.

    The LSTM receives previous closing-price values and predicts the
    following closing price.

    Parameters
    ----------
    time_step : int
        Number of previous trading days supplied to the model.

    lstm_units : int
        Number of neurons in the LSTM layer.

    Returns
    -------
    tf.keras.Model
        Uncompiled LSTM regression model.
    """

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(
                shape=(
                    time_step,
                    1,
                ),
                name="stock_sequence",
            ),

            tf.keras.layers.LSTM(
                lstm_units,
                name="lstm",
            ),

            tf.keras.layers.Dense(
                1,
                activation="linear",
                name="predicted_close",
            ),
        ],
        name=(
            f"stock_lstm_"
            f"timestep_{time_step}_"
            f"units_{lstm_units}"
        ),
    )

    return model


# ---------------------------------------------------------------------------
# 8. LSTM Model Compilation
# ---------------------------------------------------------------------------

def compile_lstm_model(
    model,
    learning_rate=LEARNING_RATE,
):
    """
    Compile an LSTM regression model.

    Mean squared error is used as the optimisation loss.

    RMSE is also recorded during fitting, although final parameter
    comparison is calculated after converting predictions back to
    their original stock-price scale.

    Parameters
    ----------
    model : tf.keras.Model
        LSTM model.

    learning_rate : float, optional
        Adam optimiser learning rate.

    Returns
    -------
    tf.keras.Model
        Compiled LSTM model.
    """

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate
    )

    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=[
            tf.keras.metrics.RootMeanSquaredError(
                name="rmse"
            )
        ],
    )

    return model


# ---------------------------------------------------------------------------
# 9. Development LSTM Smoke Test
# ---------------------------------------------------------------------------

def run_lstm_smoke_test(
    stock_code,
    training_data,
    forecast_data,
    time_step=5,
):
    """
    Run a short development training test.

    This verifies the complete TensorFlow LSTM training pipeline without
    using the final 50-day test period.

    Parameters
    ----------
    stock_code : str
        Stock ticker.

    training_data : pandas.DataFrame
        First 204 historical observations.

    forecast_data : pandas.DataFrame
        Final 50 observations.

    time_step : int, optional
        LSTM sequence length.

    Returns
    -------
    tuple
        Trained smoke-test model and Keras History object.
    """

    prepared_data = prepare_lstm_data(
        training_data,
        forecast_data,
        time_step,
    )

    (
        X_train,
        X_validation,
        y_train,
        y_validation,
    ) = split_lstm_sequences(
        prepared_data["X_train"],
        prepared_data["y_train"],
    )

    print("\nLSTM Smoke Test")
    print("===============")

    print(
        f"Stock: "
        f"{stock_code} - {STOCK_NAMES[stock_code]}"
    )

    print(
        f"Time step: "
        f"{time_step}"
    )

    print(
        f"LSTM units: "
        f"{SMOKE_TEST_LSTM_UNITS}"
    )

    print(
        f"Epochs: "
        f"{SMOKE_TEST_EPOCHS}"
    )

    print(
        f"Batch size: "
        f"{SMOKE_TEST_BATCH_SIZE}"
    )

    print(
        f"Training sequences: "
        f"{len(X_train)}"
    )

    print(
        f"Validation sequences: "
        f"{len(X_validation)}"
    )

    print(
        f"X_train shape: "
        f"{X_train.shape}"
    )

    print(
        f"X_validation shape: "
        f"{X_validation.shape}"
    )

    tf.keras.backend.clear_session()

    tf.keras.utils.set_random_seed(
        RANDOM_SEED
    )

    model = build_lstm_model(
        time_step=time_step,
        lstm_units=SMOKE_TEST_LSTM_UNITS,
    )

    model = compile_lstm_model(
        model,
        learning_rate=LEARNING_RATE,
    )

    print("\nLSTM Architecture")
    print("-----------------")

    model.summary()

    history = model.fit(
        X_train,
        y_train,
        validation_data=(
            X_validation,
            y_validation,
        ),
        epochs=SMOKE_TEST_EPOCHS,
        batch_size=SMOKE_TEST_BATCH_SIZE,
        shuffle=False,
        verbose=1,
    )

    print(
        "\nLSTM smoke test completed successfully."
    )

    return (
        model,
        history,
    )


# ---------------------------------------------------------------------------
# 10. RMSE Calculation
# ---------------------------------------------------------------------------

def calculate_rmse(
    actual_values,
    predicted_values,
):
    """
    Calculate Root Mean Squared Error.

    Parameters
    ----------
    actual_values : numpy.ndarray
        Actual stock prices.

    predicted_values : numpy.ndarray
        Predicted stock prices.

    Returns
    -------
    float
        Root Mean Squared Error.
    """

    actual_values = np.asarray(
        actual_values,
        dtype=np.float64,
    )

    predicted_values = np.asarray(
        predicted_values,
        dtype=np.float64,
    )

    squared_errors = (
        actual_values
        - predicted_values
    ) ** 2

    mean_squared_error = np.mean(
        squared_errors
    )

    return float(
        np.sqrt(
            mean_squared_error
        )
    )


# ---------------------------------------------------------------------------
# 11. Single LSTM Parameter Experiment
# ---------------------------------------------------------------------------

def run_single_lstm_experiment(
    stock_code,
    training_data,
    forecast_data,
    time_step,
    configuration,
):
    """
    Train and evaluate one LSTM parameter configuration.

    Model selection uses only the first 204 historical observations.

    A chronological validation subset is created from those historical
    sequences. The final 50 future observations remain untouched.

    Validation predictions are inverse-transformed back to actual stock
    prices so RMSE is reported in Australian dollars.

    Parameters
    ----------
    stock_code : str
        Stock ticker.

    training_data : pandas.DataFrame
        First 204 historical observations.

    forecast_data : pandas.DataFrame
        Final 50 observations reserved for later testing.

    time_step : int
        Number of previous prices supplied to the LSTM.

    configuration : dict
        Experiment configuration.

    Returns
    -------
    dict
        Experiment settings and measured results.
    """

    prepared_data = prepare_lstm_data(
        training_data,
        forecast_data,
        time_step,
    )

    (
        X_train,
        X_validation,
        y_train,
        y_validation,
    ) = split_lstm_sequences(
        prepared_data["X_train"],
        prepared_data["y_train"],
    )

    scaler = prepared_data[
        "scaler"
    ]

    tf.keras.backend.clear_session()

    tf.keras.utils.set_random_seed(
        RANDOM_SEED
    )

    model = build_lstm_model(
        time_step=time_step,
        lstm_units=configuration[
            "lstm_units"
        ],
    )

    model = compile_lstm_model(
        model,
        learning_rate=configuration[
            "learning_rate"
        ],
    )

    print("\n" + "=" * 70)

    print(
        f"{stock_code} | "
        f"Time Step {time_step} | "
        f"{configuration['name']}"
    )

    print("=" * 70)

    print(
        f"LSTM units: "
        f"{configuration['lstm_units']}"
    )

    print(
        f"Epochs: "
        f"{configuration['epochs']}"
    )

    print(
        f"Batch size: "
        f"{configuration['batch_size']}"
    )

    print(
        f"Learning rate: "
        f"{configuration['learning_rate']}"
    )

    print(
        f"Parameters: "
        f"{model.count_params():,}"
    )

    start_time = perf_counter()

    history = model.fit(
        X_train,
        y_train,
        validation_data=(
            X_validation,
            y_validation,
        ),
        epochs=configuration[
            "epochs"
        ],
        batch_size=configuration[
            "batch_size"
        ],
        shuffle=False,

        # Parameter tuning involves 36 runs.
        # Suppress per-epoch progress to keep terminal output manageable.
        verbose=0,
    )

    training_seconds = (
        perf_counter()
        - start_time
    )

    # Generate predictions for historical training and validation sequences.
    training_predictions_scaled = model.predict(
        X_train,
        verbose=0,
    )

    validation_predictions_scaled = model.predict(
        X_validation,
        verbose=0,
    )

    # Convert actual targets back to their original price scale.
    actual_training_prices = scaler.inverse_transform(
        y_train.reshape(
            -1,
            1,
        )
    ).reshape(-1)

    actual_validation_prices = scaler.inverse_transform(
        y_validation.reshape(
            -1,
            1,
        )
    ).reshape(-1)

    # Convert predicted values back to actual stock prices.
    predicted_training_prices = scaler.inverse_transform(
        training_predictions_scaled
    ).reshape(-1)

    predicted_validation_prices = scaler.inverse_transform(
        validation_predictions_scaled
    ).reshape(-1)

    training_rmse = calculate_rmse(
        actual_training_prices,
        predicted_training_prices,
    )

    validation_rmse = calculate_rmse(
        actual_validation_prices,
        predicted_validation_prices,
    )

    result = {
        "stock": stock_code,
        "stock_name": STOCK_NAMES[
            stock_code
        ],
        "time_step": time_step,
        "configuration": configuration[
            "name"
        ],
        "lstm_units": configuration[
            "lstm_units"
        ],
        "epochs": configuration[
            "epochs"
        ],
        "batch_size": configuration[
            "batch_size"
        ],
        "learning_rate": configuration[
            "learning_rate"
        ],
        "parameters": model.count_params(),
        "training_sequences": len(
            X_train
        ),
        "validation_sequences": len(
            X_validation
        ),
        "training_seconds": round(
            training_seconds,
            2,
        ),
        "final_training_loss": round(
            float(
                history.history[
                    "loss"
                ][-1]
            ),
            6,
        ),
        "final_validation_loss": round(
            float(
                history.history[
                    "val_loss"
                ][-1]
            ),
            6,
        ),
        "final_training_scaled_rmse": round(
            float(
                history.history[
                    "rmse"
                ][-1]
            ),
            6,
        ),
        "final_validation_scaled_rmse": round(
            float(
                history.history[
                    "val_rmse"
                ][-1]
            ),
            6,
        ),
        "training_rmse_dollars": round(
            training_rmse,
            4,
        ),
        "validation_rmse_dollars": round(
            validation_rmse,
            4,
        ),
    }

    print(
        f"Training RMSE: "
        f"${training_rmse:.4f}"
    )

    print(
        f"Validation RMSE: "
        f"${validation_rmse:.4f}"
    )

    print(
        f"Final training loss: "
        f"{result['final_training_loss']:.6f}"
    )

    print(
        f"Final validation loss: "
        f"{result['final_validation_loss']:.6f}"
    )

    print(
        f"Training time: "
        f"{training_seconds:.2f} seconds"
    )

    return result


# ---------------------------------------------------------------------------
# 12. LSTM Parameter Experiment Runner
# ---------------------------------------------------------------------------

def run_lstm_parameter_experiments(
    stock_datasets,
):
    """
    Run all required LSTM neuron/epoch experiments.

    The experiment matrix contains:

    3 stocks
        RIO
        BHP
        FMG

    x 2 time steps
        5
        10

    x 6 parameter configurations

    = 36 experiment runs.

    Parameters
    ----------
    stock_datasets : dict
        Loaded mining-group datasets.

    Returns
    -------
    pandas.DataFrame
        Results for all 36 controlled experiments.
    """

    results = []

    total_runs = (
        len(STOCK_FILES)
        * len(TIME_STEPS)
        * len(LSTM_CONFIGURATIONS)
    )

    current_run = 0

    print("\nLSTM Parameter Experiments")
    print("==========================")

    print(
        f"Stocks: "
        f"{len(STOCK_FILES)}"
    )

    print(
        f"Time steps: "
        f"{len(TIME_STEPS)}"
    )

    print(
        f"Configurations: "
        f"{len(LSTM_CONFIGURATIONS)}"
    )

    print(
        f"Total experiment runs: "
        f"{total_runs}"
    )

    for stock_code, datasets in stock_datasets.items():

        for time_step in TIME_STEPS:

            for configuration in LSTM_CONFIGURATIONS:

                current_run += 1

                print(
                    f"\nExperiment "
                    f"{current_run}/{total_runs}"
                )

                result = run_single_lstm_experiment(
                    stock_code=stock_code,
                    training_data=datasets[
                        "training"
                    ],
                    forecast_data=datasets[
                        "forecast"
                    ],
                    time_step=time_step,
                    configuration=configuration,
                )

                results.append(
                    result
                )

    results_table = pd.DataFrame(
        results
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / "lstm_parameter_experiment_results.csv"
    )

    results_table.to_csv(
        output_path,
        index=False,
    )

    print("\nLSTM Parameter Experiment Results")
    print("=================================")

    print(
        results_table.to_string(
            index=False
        )
    )

    print(
        f"\nResults saved to: "
        f"{output_path}"
    )

    return results_table


# ---------------------------------------------------------------------------
# 13. Best LSTM Configuration Summary
# ---------------------------------------------------------------------------

def display_best_lstm_configurations(
    results_table,
):
    """
    Display the lowest-validation-RMSE configuration for each stock and
    each assessment-required time step.

    Parameters
    ----------
    results_table : pandas.DataFrame
        Complete LSTM parameter experiment results.
    """

    print("\nBest LSTM Configurations")
    print("========================")

    for stock_code in STOCK_FILES:

        for time_step in TIME_STEPS:

            subset = results_table[
                (
                    results_table[
                        "stock"
                    ]
                    == stock_code
                )
                &
                (
                    results_table[
                        "time_step"
                    ]
                    == time_step
                )
            ]

            if subset.empty:
                continue

            best_index = (
                subset[
                    "validation_rmse_dollars"
                ]
                .idxmin()
            )

            best_result = results_table.loc[
                best_index
            ]

            print(
                f"\n{stock_code} "
                f"- Time Step {time_step}"
            )

            print(
                f"Configuration: "
                f"{best_result['configuration']}"
            )

            print(
                f"LSTM units: "
                f"{int(best_result['lstm_units'])}"
            )

            print(
                f"Epochs: "
                f"{int(best_result['epochs'])}"
            )

            print(
                f"Training RMSE: "
                f"${best_result['training_rmse_dollars']:.4f}"
            )

            print(
                f"Validation RMSE: "
                f"${best_result['validation_rmse_dollars']:.4f}"
            )



# ---------------------------------------------------------------------------
# 14. Selected Configuration Lookup
# ---------------------------------------------------------------------------

def get_lstm_configuration(
    configuration_name,
):
    """
    Return one LSTM experiment configuration by name.

    Parameters
    ----------
    configuration_name : str
        Configuration name stored in SELECTED_LSTM_CONFIGURATIONS.

    Returns
    -------
    dict
        Matching LSTM experiment configuration.

    Raises
    ------
    ValueError
        If the requested configuration does not exist.
    """

    for configuration in LSTM_CONFIGURATIONS:

        if configuration["name"] == configuration_name:
            return configuration

    raise ValueError(
        f"Unknown LSTM configuration: {configuration_name}"
    )


# ---------------------------------------------------------------------------
# 15. Recursive 50-Day Forecasting
# ---------------------------------------------------------------------------

def recursive_stock_forecast(
    model,
    scaled_training_prices,
    time_step,
    forecast_days=FORECAST_DAYS,
):
    """
    Produce a recursive multi-day stock-price forecast.

    The final historical time-step values form the initial input window.
    Each predicted value is appended to the rolling window and becomes
    part of the input used to predict the following day.

    Actual future prices are never supplied to the model during recursive
    forecasting.

    Parameters
    ----------
    model : tf.keras.Model
        Trained final LSTM model.

    scaled_training_prices : numpy.ndarray
        Scaled historical closing prices from the first 204 observations.

    time_step : int
        Number of previous prices supplied to the LSTM.

    forecast_days : int, optional
        Number of future trading days to predict.

    Returns
    -------
    numpy.ndarray
        Recursive predictions in the scaled domain with shape
        (forecast_days, 1).
    """

    if len(scaled_training_prices) < time_step:
        raise ValueError(
            "Insufficient historical prices for recursive forecasting."
        )

    forecast_window = (
        scaled_training_prices[
            -time_step:
        ]
        .reshape(-1)
        .tolist()
    )

    predictions = []

    for _ in range(forecast_days):

        model_input = np.asarray(
            forecast_window,
            dtype=np.float32,
        ).reshape(
            1,
            time_step,
            1,
        )

        predicted_value = float(
            model.predict(
                model_input,
                verbose=0,
            )[0, 0]
        )

        predictions.append(
            predicted_value
        )

        # Use the model prediction, not the actual future price, when
        # preparing the next recursive input window.
        forecast_window.append(
            predicted_value
        )

        forecast_window = forecast_window[
            -time_step:
        ]

    return np.asarray(
        predictions,
        dtype=np.float32,
    ).reshape(
        -1,
        1,
    )


# ---------------------------------------------------------------------------
# 16. Final LSTM Visualisations
# ---------------------------------------------------------------------------

def save_lstm_loss_plot(
    history,
    stock_code,
    time_step,
):
    """
    Save training and validation loss curves for a selected LSTM model.

    Parameters
    ----------
    history : tf.keras.callbacks.History
        Training history from the chronological train/validation stage.

    stock_code : str
        Stock ticker.

    time_step : int
        LSTM sequence length.

    Returns
    -------
    pathlib.Path
        Saved loss-curve path.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    epochs = range(
        1,
        len(history.history["loss"]) + 1,
    )

    figure = plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        epochs,
        history.history["loss"],
        label="Training Loss",
    )

    plt.plot(
        epochs,
        history.history["val_loss"],
        label="Validation Loss",
    )

    plt.title(
        f"{stock_code} LSTM Loss - Time Step {time_step}"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Mean Squared Error"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / f"lstm_{stock_code}_t{time_step}_loss.png"
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    print(
        f"Loss curve saved to: "
        f"{output_path}"
    )

    return output_path


def save_stock_forecast_plot(
    stock_code,
    stock_data,
    forecast_data,
    predicted_prices,
    time_step,
):
    """
    Plot the observed stock-price data and the final 50-day forecast.

    Both individual data points and fitted curves are displayed to satisfy
    the assessment requirement for original-versus-predicted stock-price
    visualisation.

    Parameters
    ----------
    stock_code : str
        Stock ticker.

    stock_data : pandas.DataFrame
        Complete 254-day chronological stock dataset.

    forecast_data : pandas.DataFrame
        Final 50 held-out observations.

    predicted_prices : numpy.ndarray
        Predicted prices for the final 50 trading days.

    time_step : int
        LSTM time-step setting.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure = plt.figure(
        figsize=(12, 7)
    )

    # ---------------------------------------------------------------
    # Complete observed stock-price series
    # ---------------------------------------------------------------

    plt.plot(
        stock_data[DATE_COLUMN],
        stock_data[CLOSE_COLUMN],
        label="Actual Closing Price",
        linewidth=1.5,
    )

    # Display the individual observed data points.
    plt.scatter(
        stock_data[DATE_COLUMN],
        stock_data[CLOSE_COLUMN],
        s=12,
        label="Actual Data Points",
    )

    # ---------------------------------------------------------------
    # Final 50-day predicted series
    # ---------------------------------------------------------------

    plt.plot(
        forecast_data[DATE_COLUMN],
        predicted_prices,
        label="Predicted Closing Price",
        linewidth=2.0,
    )

    # Display individual predicted values.
    plt.scatter(
        forecast_data[DATE_COLUMN],
        predicted_prices,
        s=20,
        label="Predicted Data Points",
    )

    # ---------------------------------------------------------------
    # Mark where forecasting begins
    # ---------------------------------------------------------------

    forecast_start_date = (
        forecast_data[
            DATE_COLUMN
        ].iloc[0]
    )

    plt.axvline(
        forecast_start_date,
        linestyle="--",
        label="Forecast Start",
    )

    plt.title(
        f"{stock_code} Stock Price Forecast "
        f"(Time Step {time_step})"
    )

    plt.xlabel(
        "Date"
    )

    plt.ylabel(
        "Closing Price ($)"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    figure.autofmt_xdate()

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / f"lstm_{stock_code}_t{time_step}_forecast.png"
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    print(
        f"Forecast plot saved to: "
        f"{output_path}"
    )

    return output_path


# ---------------------------------------------------------------------------
# 17. Final Selected LSTM Training and Forecasting
# ---------------------------------------------------------------------------

def run_final_lstm_forecast(
    stock_code,
    datasets,
    time_step,
):
    """
    Train the selected model and forecast the final 50 trading days.

    Model selection has already been completed using chronological
    validation results from the parameter-tuning stage.

    Two training stages are used here:

    1. A chronological training/validation run is repeated to generate
       the required training and validation loss curves.
    2. A fresh model is then trained using all historical sequences from
       the first 204 observations before the final recursive forecast.

    The final 50 held-out closing prices are used only after forecasting
    to calculate the final test RMSE.

    Parameters
    ----------
    stock_code : str
        Stock ticker.

    datasets : dict
        Complete, historical training, and held-out forecast datasets.

    time_step : int
        LSTM sequence length.

    Returns
    -------
    dict
        Final model settings and 50-day forecast results.
    """

    configuration_name = (
        SELECTED_LSTM_CONFIGURATIONS[
            (
                stock_code,
                time_step,
            )
        ]
    )

    configuration = get_lstm_configuration(
        configuration_name
    )

    training_data = datasets[
        "training"
    ]

    forecast_data = datasets[
        "forecast"
    ]

    stock_data = datasets[
        "all"
    ]

    prepared_data = prepare_lstm_data(
        training_data,
        forecast_data,
        time_step,
    )

    scaler = prepared_data[
        "scaler"
    ]

    X_all = prepared_data[
        "X_train"
    ]

    y_all = prepared_data[
        "y_train"
    ]

    (
        X_train,
        X_validation,
        y_train,
        y_validation,
    ) = split_lstm_sequences(
        X_all,
        y_all,
    )

    print("\n" + "=" * 70)

    print(
        f"Final LSTM Forecast: "
        f"{stock_code} | Time Step {time_step}"
    )

    print("=" * 70)

    print(
        f"Configuration: "
        f"{configuration_name}"
    )

    print(
        f"LSTM units: "
        f"{configuration['lstm_units']}"
    )

    print(
        f"Epochs: "
        f"{configuration['epochs']}"
    )

    print(
        f"Batch size: "
        f"{configuration['batch_size']}"
    )

    print(
        f"Learning rate: "
        f"{configuration['learning_rate']}"
    )

    print(
        f"Historical sequences: "
        f"{len(X_all)}"
    )

    # ---------------------------------------------------------------
    # Stage 1: chronological train/validation evidence
    # ---------------------------------------------------------------

    tf.keras.backend.clear_session()

    tf.keras.utils.set_random_seed(
        RANDOM_SEED
    )

    validation_model = build_lstm_model(
        time_step=time_step,
        lstm_units=configuration[
            "lstm_units"
        ],
    )

    validation_model = compile_lstm_model(
        validation_model,
        learning_rate=configuration[
            "learning_rate"
        ],
    )

    validation_history = validation_model.fit(
        X_train,
        y_train,
        validation_data=(
            X_validation,
            y_validation,
        ),
        epochs=configuration[
            "epochs"
        ],
        batch_size=configuration[
            "batch_size"
        ],
        shuffle=False,
        verbose=0,
    )

    save_lstm_loss_plot(
        validation_history,
        stock_code,
        time_step,
    )

    # ---------------------------------------------------------------
    # Stage 2: train final model on all historical sequences
    # ---------------------------------------------------------------

    tf.keras.backend.clear_session()

    tf.keras.utils.set_random_seed(
        RANDOM_SEED
    )

    final_model = build_lstm_model(
        time_step=time_step,
        lstm_units=configuration[
            "lstm_units"
        ],
    )

    final_model = compile_lstm_model(
        final_model,
        learning_rate=configuration[
            "learning_rate"
        ],
    )

    print(
        f"Training final model using "
        f"{len(X_all)} historical sequences..."
    )

    training_start = perf_counter()

    final_model.fit(
        X_all,
        y_all,
        epochs=configuration[
            "epochs"
        ],
        batch_size=configuration[
            "batch_size"
        ],
        shuffle=False,
        verbose=0,
    )

    training_seconds = (
        perf_counter()
        - training_start
    )

    # ---------------------------------------------------------------
    # Stage 3: recursive forecasting
    # ---------------------------------------------------------------

    predictions_scaled = recursive_stock_forecast(
        final_model,
        prepared_data[
            "scaled_training"
        ],
        time_step=time_step,
        forecast_days=FORECAST_DAYS,
    )

    predicted_prices = (
        scaler.inverse_transform(
            predictions_scaled
        )
        .reshape(-1)
    )

    actual_prices = (
        forecast_data[
            CLOSE_COLUMN
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    forecast_rmse = calculate_rmse(
        actual_prices,
        predicted_prices,
    )

    print(
        f"Final 50-day forecast RMSE: "
        f"${forecast_rmse:.4f}"
    )

    # ---------------------------------------------------------------
    # Save prediction table
    # ---------------------------------------------------------------

    absolute_errors = np.abs(
        actual_prices
        - predicted_prices
    )

    forecast_results = pd.DataFrame(
        {
            "date": (
                forecast_data[
                    DATE_COLUMN
                ]
                .dt.strftime(
                    "%d/%m/%Y"
                )
                .to_numpy()
            ),
            "actual_close": np.round(
                actual_prices,
                4,
            ),
            "predicted_close": np.round(
                predicted_prices,
                4,
            ),
            "absolute_error": np.round(
                absolute_errors,
                4,
            ),
        }
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    forecast_csv_path = (
        OUTPUT_DIR
        / (
            f"lstm_{stock_code}_"
            f"t{time_step}_forecast.csv"
        )
    )

    forecast_results.to_csv(
        forecast_csv_path,
        index=False,
    )

    print(
        f"Forecast data saved to: "
        f"{forecast_csv_path}"
    )

    # ---------------------------------------------------------------
    # Save final figures and model
    # ---------------------------------------------------------------

    save_stock_forecast_plot(
        stock_code,
        stock_data,
        forecast_data,
        predicted_prices,
        time_step,
    )

    model_path = (
        OUTPUT_DIR
        / (
            f"lstm_{stock_code}_"
            f"t{time_step}.keras"
        )
    )

    final_model.save(
        model_path
    )

    print(
        f"Final model saved to: "
        f"{model_path}"
    )

    return {
        "stock": stock_code,
        "stock_name": STOCK_NAMES[
            stock_code
        ],
        "time_step": time_step,
        "configuration": configuration_name,
        "lstm_units": configuration[
            "lstm_units"
        ],
        "epochs": configuration[
            "epochs"
        ],
        "batch_size": configuration[
            "batch_size"
        ],
        "learning_rate": configuration[
            "learning_rate"
        ],
        "parameters": final_model.count_params(),
        "historical_sequences": len(
            X_all
        ),
        "forecast_days": FORECAST_DAYS,
        "forecast_rmse_dollars": round(
            forecast_rmse,
            4,
        ),
        "mean_absolute_error_dollars": round(
            float(
                np.mean(
                    absolute_errors
                )
            ),
            4,
        ),
        "training_seconds": round(
            training_seconds,
            2,
        ),
    }


# ---------------------------------------------------------------------------
# 18. Final Forecast Runner
# ---------------------------------------------------------------------------

def run_all_final_lstm_forecasts(
    stock_datasets,
):
    """
    Run all six selected stock/time-step final forecasts.

    Returns
    -------
    pandas.DataFrame
        Final comparison results.
    """

    final_results = []

    total_runs = (
        len(STOCK_FILES)
        * len(TIME_STEPS)
    )

    current_run = 0

    print("\nFinal LSTM 50-Day Forecasts")
    print("===========================")

    print(
        f"Total final models: "
        f"{total_runs}"
    )

    for stock_code, datasets in stock_datasets.items():

        for time_step in TIME_STEPS:

            current_run += 1

            print(
                f"\nFinal forecast "
                f"{current_run}/{total_runs}"
            )

            result = run_final_lstm_forecast(
                stock_code=stock_code,
                datasets=datasets,
                time_step=time_step,
            )

            final_results.append(
                result
            )

    results_table = pd.DataFrame(
        final_results
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / "lstm_final_forecast_results.csv"
    )

    results_table.to_csv(
        output_path,
        index=False,
    )

    print("\nFinal LSTM Forecast Results")
    print("===========================")

    print(
        results_table.to_string(
            index=False
        )
    )

    print(
        f"\nFinal comparison saved to: "
        f"{output_path}"
    )

    return results_table


# ---------------------------------------------------------------------------
# 19. LSTM Data Preparation Summary
# ---------------------------------------------------------------------------

def display_lstm_preparation_summary(
    stock_code,
    training_data,
    prepared_data,
    time_step,
):
    """
    Display scaling information and LSTM tensor shapes.
    """

    scaled_training = prepared_data[
        "scaled_training"
    ]

    scaled_forecast = prepared_data[
        "scaled_forecast"
    ]

    X_sequences = prepared_data[
        "X_train"
    ]

    y_sequences = prepared_data[
        "y_train"
    ]

    expected_sequences = (
        len(training_data)
        - time_step
    )

    print(
        f"\n{stock_code} - Time Step {time_step}"
    )

    print(
        "-" * 40
    )

    print(
        f"Training observations: "
        f"{len(training_data)}"
    )

    print(
        f"Expected sequences: "
        f"{expected_sequences}"
    )

    print(
        f"Generated sequences: "
        f"{len(X_sequences)}"
    )

    print(
        f"X_train shape: "
        f"{X_sequences.shape}"
    )

    print(
        f"y_train shape: "
        f"{y_sequences.shape}"
    )

    print(
        f"Training scaled minimum: "
        f"{scaled_training.min():.4f}"
    )

    print(
        f"Training scaled maximum: "
        f"{scaled_training.max():.4f}"
    )

    print(
        f"Forecast scaled minimum: "
        f"{scaled_forecast.min():.4f}"
    )

    print(
        f"Forecast scaled maximum: "
        f"{scaled_forecast.max():.4f}"
    )

    if len(X_sequences) != expected_sequences:
        raise ValueError(
            f"{stock_code}: unexpected number "
            "of generated sequences."
        )


# ---------------------------------------------------------------------------
# 20. Dataset Summary
# ---------------------------------------------------------------------------

def display_stock_summary(
    stock_code,
    stock_data,
    training_data,
    forecast_data,
):
    """
    Display preprocessing information for one stock.
    """

    stock_name = STOCK_NAMES[
        stock_code
    ]

    print(
        "\n"
        + "=" * 60
    )

    print(
        f"{stock_code} - {stock_name}"
    )

    print(
        "=" * 60
    )

    print(
        f"Total records: "
        f"{len(stock_data)}"
    )

    if len(stock_data) == EXPECTED_RECORDS:
        print(
            f"Expected record count: "
            f"{EXPECTED_RECORDS} [OK]"
        )
    else:
        print(
            f"Expected record count: "
            f"{EXPECTED_RECORDS} "
            f"[Found {len(stock_data)}]"
        )

    print(
        f"Missing values: "
        f"{stock_data.isna().sum().sum()}"
    )

    print(
        f"Duplicate dates: "
        f"{stock_data[DATE_COLUMN].duplicated().sum()}"
    )

    print("\nChronological range:")

    print(
        f"First date: "
        f"{stock_data[DATE_COLUMN].iloc[0].strftime('%d/%m/%Y')}"
    )

    print(
        f"Last date: "
        f"{stock_data[DATE_COLUMN].iloc[-1].strftime('%d/%m/%Y')}"
    )

    print("\nClosing price range:")

    print(
        f"Minimum close: "
        f"${stock_data[CLOSE_COLUMN].min():.2f}"
    )

    print(
        f"Maximum close: "
        f"${stock_data[CLOSE_COLUMN].max():.2f}"
    )

    print(
        f"Average close: "
        f"${stock_data[CLOSE_COLUMN].mean():.2f}"
    )

    print("\nTraining period:")

    print(
        f"Records: "
        f"{len(training_data)}"
    )

    print(
        f"From: "
        f"{training_data[DATE_COLUMN].iloc[0].strftime('%d/%m/%Y')}"
    )

    print(
        f"To: "
        f"{training_data[DATE_COLUMN].iloc[-1].strftime('%d/%m/%Y')}"
    )

    print("\nForecast/testing period:")

    print(
        f"Records: "
        f"{len(forecast_data)}"
    )

    print(
        f"From: "
        f"{forecast_data[DATE_COLUMN].iloc[0].strftime('%d/%m/%Y')}"
    )

    print(
        f"To: "
        f"{forecast_data[DATE_COLUMN].iloc[-1].strftime('%d/%m/%Y')}"
    )


# ---------------------------------------------------------------------------
# 21. Assigned Mining Group Loader
# ---------------------------------------------------------------------------

def load_assigned_stocks():
    """
    Load, validate, order, and split all three assigned mining stocks.

    Returns
    -------
    dict
        Complete, historical training, and future forecast datasets.
    """

    stock_datasets = {}

    for stock_code, file_path in STOCK_FILES.items():

        stock_data = load_stock_data(
            stock_code,
            file_path,
        )

        training_data, forecast_data = split_stock_data(
            stock_data
        )

        stock_datasets[
            stock_code
        ] = {
            "all": stock_data,
            "training": training_data,
            "forecast": forecast_data,
        }

    return stock_datasets



# ---------------------------------------------------------------------------
# Assessment Menu
# ---------------------------------------------------------------------------

def display_lstm_menu():
    """Display the Task 2.2 menu required by the assessment."""

    print(
        "\nCOIT29225 Assessment 3 - LSTM Stock Forecasting"
    )

    print(
        "================================================"
    )

    print(
        "1. Time step = 5"
    )

    print(
        "2. Time step = 10"
    )

    print(
        "3. Quit"
    )


def run_forecasts_for_time_step(
    stock_datasets,
    time_step,
):
    """
    Run final forecasts for all three assigned stocks using one time step.

    Parameters
    ----------
    stock_datasets : dict
        Loaded RIO, BHP, and FMG datasets.

    time_step : int
        Assessment time step, either 5 or 10.

    Returns
    -------
    pandas.DataFrame
        Final results for the three stock forecasts.
    """

    results = []

    print(
        f"\nRunning final LSTM forecasts "
        f"with time step {time_step}"
    )

    print(
        "=" * 55
    )

    for stock_code, datasets in stock_datasets.items():

        result = run_final_lstm_forecast(
            stock_code,
            datasets,
            time_step,
        )

        results.append(
            result
        )

    results_table = pd.DataFrame(
        results
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / f"lstm_t{time_step}_final_results.csv"
    )

    results_table.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nTime Step {time_step} Forecast Results"
    )

    print(
        "=" * 40
    )

    print(
        results_table[
            [
                "stock",
                "configuration",
                "lstm_units",
                "epochs",
                "forecast_rmse_dollars",
                "mean_absolute_error_dollars",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        f"\nResults saved to: "
        f"{output_path}"
    )

    return results_table


def run_lstm_menu(
    stock_datasets,
):
    """
    Run the assessment-required Task 2.2 menu.
    """

    while True:

        display_lstm_menu()

        choice = input(
            "\nSelect an option (1-3): "
        ).strip()

        if choice == "1":

            run_forecasts_for_time_step(
                stock_datasets,
                time_step=5,
            )

        elif choice == "2":

            run_forecasts_for_time_step(
                stock_datasets,
                time_step=10,
            )

        elif choice == "3":

            print(
                "\nExiting application."
            )

            break

        else:

            print(
                "\nInvalid selection. "
                "Please enter 1, 2, or 3."
            )


# ---------------------------------------------------------------------------
# Main Program
# ---------------------------------------------------------------------------

def main():
    """
    Load the assigned stock datasets and launch the Task 2.2 menu.
    """

    print(
        "\nLoading assigned mining stock datasets..."
    )

    try:

        stock_datasets = load_assigned_stocks()

    except (
        FileNotFoundError,
        ValueError,
    ) as error:

        print(
            f"\nData loading error: "
            f"{error}"
        )

        return

    print(
        f"Loaded {len(stock_datasets)} stocks successfully."
    )

    print(
        "Stocks: RIO, BHP, FMG"
    )

    print(
        "Historical period: 204 trading days"
    )

    print(
        "Forecast/testing period: 50 trading days"
    )

    run_lstm_menu(
        stock_datasets
    )


if __name__ == "__main__":
    main()