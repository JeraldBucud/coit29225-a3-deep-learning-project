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

The program therefore:

1. Loads the assigned stock CSV files.
2. Extracts the Date and Close $ columns.
3. Validates dates and closing prices.
4. Converts the data into chronological order.
5. Uses the first 204 observations as the training period.
6. Uses the remaining 50 observations as the forecasting/testing period.

LSTM modelling, time-step experiments, parameter tuning, RMSE
evaluation, and visualisation are added in later development stages.
"""

from pathlib import Path

import pandas as pd


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

# Human-readable names used in program output and later report figures.
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
# 4. Dataset Summary
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
# 5. Assigned Mining Group Loader
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
# 6. Main Program
# ---------------------------------------------------------------------------

def main():
    """
    Run the Task 2.2 stock-data preprocessing checkpoint.
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


if __name__ == "__main__":
    main()