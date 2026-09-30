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

The supplied CSV files contain approximately 254 trading days and are
stored in reverse chronological order, from the newest date to the
oldest date.

The program currently performs the following stages:

1. Loads the assigned stock CSV files.
2. Extracts the Date and Close $ columns.
3. Validates dates and closing prices.
4. Converts the data into chronological order.
5. Uses the first 204 observations as the historical training period.
6. Uses the remaining 50 observations as the forecasting/testing period.
7. Scales closing prices using training data only.
8. Generates LSTM sequences for time steps 5 and 10.
9. Creates chronological training and validation subsets.
10. Builds and compiles a basic LSTM regression model.
11. Runs a short smoke test to validate the LSTM training pipeline.

Parameter tuning, 50-day recursive forecasting, RMSE comparison,
visualisation, and the final assessment menu are added in later stages.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------

# Directory containing the supplied Assessment 3 stock-price CSV files.
DATA_DIR = Path("data")

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

# The supplied dataset contains approximately one trading year.
EXPECTED_RECORDS = 254

# Assessment-required chronological split.
TRAINING_DAYS = 204
FORECAST_DAYS = 50

# Assessment-required LSTM time-step settings.
TIME_STEPS = (
    5,
    10,
)

# Fraction of the generated training sequences reserved for
# chronological validation.
VALIDATION_FRACTION = 0.20

# Fixed random seed used to improve experiment reproducibility.
RANDOM_SEED = 42

# Development-only LSTM smoke-test settings.
SMOKE_TEST_LSTM_UNITS = 32
SMOKE_TEST_EPOCHS = 2
SMOKE_TEST_BATCH_SIZE = 16

# Initial optimiser learning rate.
LEARNING_RATE = 0.001


# ---------------------------------------------------------------------------
# 2. Stock Data Loading and Validation
# ---------------------------------------------------------------------------

def load_stock_data(stock_code, file_path):
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

    # Load the complete CSV first because RIO and FMG contain additional
    # market columns while BHP contains only Date and Close $.
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

    # Retain only the variables required for this forecasting task.
    stock_data = raw_data[
        [
            DATE_COLUMN,
            CLOSE_COLUMN,
        ]
    ].copy()

    # Parse dates using Australian day/month/year ordering.
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

    # Invalid values should be reported rather than silently removed.
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

    # Every trading date should appear only once.
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

    # The supplied files are newest-to-oldest. Sorting ascending converts
    # them into chronological order before any time-series modelling.
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

def split_stock_data(stock_data):
    """
    Divide one chronological stock series into training and forecast sets.

    The first 204 trading days are used as the historical training period.
    The final 50 trading days are reserved as the forecasting/testing period.

    Parameters
    ----------
    stock_data : pandas.DataFrame
        Chronologically ordered stock-price data.

    Returns
    -------
    tuple
        Training DataFrame and forecasting/testing DataFrame.

    Raises
    ------
    ValueError
        If there are insufficient records for the required split.
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
    Scale stock closing prices using training data only.

    MinMaxScaler is fitted exclusively on the first 204 training
    observations. This prevents information from the 50 future
    observations leaking into the model preparation process.

    Parameters
    ----------
    training_data : pandas.DataFrame
        First 204 chronological stock observations.

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

    # Fit only on historical training values.
    scaled_training_prices = scaler.fit_transform(
        training_prices
    )

    # Apply the fitted training scaler to future observations.
    #
    # Values may be below 0 or above 1 when future prices fall outside
    # the historical range. This is valid and avoids future-data leakage.
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

    For example, with a time step of 5:

        [day 1, day 2, day 3, day 4, day 5] -> day 6
        [day 2, day 3, day 4, day 5, day 6] -> day 7

    Parameters
    ----------
    scaled_prices : numpy.ndarray
        Scaled closing-price values with shape (observations, 1).

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
    Prepare scaled stock prices and LSTM sequences for one time step.

    Parameters
    ----------
    training_data : pandas.DataFrame
        First 204 historical observations.

    forecast_data : pandas.DataFrame
        Final 50 observations reserved for forecasting/testing.

    time_step : int
        Number of historical values supplied to the LSTM.

    Returns
    -------
    dict
        Fitted scaler, scaled datasets, and LSTM training arrays.
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
    Split LSTM sequences chronologically into training and validation sets.

    Time-series data must not be randomly shuffled because later
    observations should not leak into earlier training examples.

    Parameters
    ----------
    X : numpy.ndarray
        Complete LSTM sequence tensor generated from the 204-day
        historical training period.

    y : numpy.ndarray
        Corresponding target values.

    validation_fraction : float, optional
        Fraction of sequences reserved for validation.

    Returns
    -------
    tuple
        X_train, X_validation, y_train, y_validation.

    Raises
    ------
    ValueError
        If the validation fraction or resulting split is invalid.
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

    # Preserve chronological ordering. Earlier sequences are used for
    # model fitting and later sequences are used for validation.
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
    lstm_units=SMOKE_TEST_LSTM_UNITS,
):
    """
    Build a simple LSTM regression model.

    The LSTM receives a sequence of previous closing prices and predicts
    the next scaled closing price.

    Parameters
    ----------
    time_step : int
        Number of previous trading days supplied to the model.

    lstm_units : int, optional
        Number of neurons in the LSTM layer.

    Returns
    -------
    tf.keras.Model
        Uncompiled LSTM model.
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
    Compile an LSTM model for stock-price regression.

    Mean squared error is used as the optimisation loss while RMSE is
    recorded as an evaluation metric because the assessment requires
    root mean-squared error results.

    Parameters
    ----------
    model : tf.keras.Model
        LSTM model to compile.

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
# 9. LSTM Training Smoke Test
# ---------------------------------------------------------------------------

def run_lstm_smoke_test(
    stock_code,
    training_data,
    forecast_data,
    time_step=5,
):
    """
    Run a short LSTM training test.

    This function verifies that the complete preprocessing, sequence,
    model-building, training, and validation pipeline works correctly.

    The final 50-day forecasting/testing dataset is not used for model
    fitting or validation.

    Parameters
    ----------
    stock_code : str
        Stock ticker used for the smoke test.

    training_data : pandas.DataFrame
        First 204 historical observations.

    forecast_data : pandas.DataFrame
        Final 50 future observations.

    time_step : int, optional
        LSTM input sequence length.

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
        f"y_train shape: "
        f"{y_train.shape}"
    )

    print(
        f"X_validation shape: "
        f"{X_validation.shape}"
    )

    print(
        f"y_validation shape: "
        f"{y_validation.shape}"
    )

    # Clear any previously constructed Keras model.
    tf.keras.backend.clear_session()

    # Use a fixed random seed for reproducible development experiments.
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

        # Do not randomly reorder time-series sequences during fitting.
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
# 10. LSTM Data Preparation Summary
# ---------------------------------------------------------------------------

def display_lstm_preparation_summary(
    stock_code,
    training_data,
    prepared_data,
    time_step,
):
    """
    Display scaling information and tensor shapes prepared for the LSTM.

    Parameters
    ----------
    stock_code : str
        Stock ticker.

    training_data : pandas.DataFrame
        First 204 stock observations.

    prepared_data : dict
        Output returned by prepare_lstm_data().

    time_step : int
        Number of previous days in each LSTM input sequence.
    """

    scaled_training = prepared_data[
        "scaled_training"
    ]

    scaled_forecast = prepared_data[
        "scaled_forecast"
    ]

    X_train = prepared_data[
        "X_train"
    ]

    y_train = prepared_data[
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
        f"{len(X_train)}"
    )

    print(
        f"X_train shape: "
        f"{X_train.shape}"
    )

    print(
        f"y_train shape: "
        f"{y_train.shape}"
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

    if len(X_train) != expected_sequences:
        raise ValueError(
            f"{stock_code}: unexpected number "
            "of generated sequences."
        )


# ---------------------------------------------------------------------------
# 11. Dataset Summary
# ---------------------------------------------------------------------------

def display_stock_summary(
    stock_code,
    stock_data,
    training_data,
    forecast_data,
):
    """
    Display preprocessing information for one stock.

    Parameters
    ----------
    stock_code : str
        Stock ticker.

    stock_data : pandas.DataFrame
        Complete chronological stock dataset.

    training_data : pandas.DataFrame
        First 204 trading-day observations.

    forecast_data : pandas.DataFrame
        Final 50 trading-day observations.
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
# 12. Assigned Mining Group Loader
# ---------------------------------------------------------------------------

def load_assigned_stocks():
    """
    Load, validate, order, and split all three assigned mining stocks.

    Returns
    -------
    dict
        Dictionary containing complete, training, and forecast datasets
        for RIO, BHP, and FMG.
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
# 13. Main Program
# ---------------------------------------------------------------------------

def main():
    """
    Run the current Task 2.2 LSTM development pipeline.

    The current checkpoint validates:

    - stock-data loading;
    - chronological ordering;
    - 204/50 splitting;
    - MinMax scaling;
    - time-step sequence preparation;
    - chronological training/validation splitting;
    - basic LSTM construction;
    - basic LSTM model training.
    """

    print(
        "\nCOIT29225 Assessment 3 - Task 2.2"
    )

    print(
        "Australian Stock Price Forecasting"
    )

    print(
        "Assigned Group: Mining"
    )

    print(
        "Stocks: RIO, BHP, FMG"
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

    # ------------------------------------------------------------------
    # Dataset summaries
    # ------------------------------------------------------------------

    for stock_code, datasets in stock_datasets.items():

        display_stock_summary(
            stock_code,
            datasets["all"],
            datasets["training"],
            datasets["forecast"],
        )

    print(
        "\nAll assigned stock datasets "
        "were loaded successfully."
    )

    print(
        "Chronological preprocessing "
        "and 204/50 splitting complete."
    )

    # ------------------------------------------------------------------
    # Verify LSTM scaling and sequence preparation
    # ------------------------------------------------------------------

    print(
        "\nLSTM Sequence Preparation"
    )

    print(
        "========================="
    )

    for stock_code, datasets in stock_datasets.items():

        for time_step in TIME_STEPS:

            prepared_data = prepare_lstm_data(
                datasets["training"],
                datasets["forecast"],
                time_step,
            )

            display_lstm_preparation_summary(
                stock_code,
                datasets["training"],
                prepared_data,
                time_step,
            )

    print(
        "\nLSTM scaling and sequence "
        "preparation complete."
    )

    # ------------------------------------------------------------------
    # Development LSTM smoke test
    # ------------------------------------------------------------------
    #
    # Run only one small model before implementing the complete parameter
    # experiments. RIO with time step 5 is sufficient to verify that the
    # full Keras training and validation pipeline works.

    run_lstm_smoke_test(
        stock_code="RIO",
        training_data=stock_datasets[
            "RIO"
        ]["training"],
        forecast_data=stock_datasets[
            "RIO"
        ]["forecast"],
        time_step=5,
    )


if __name__ == "__main__":
    main()