# COIT29225 Assessment 3 - Neural Networks and Deep Learning

CQUniversity  
COIT29225 Neural Networks & Deep Learning  
Term 2, 2026

## Overview

This repository contains the development work for COIT29225 Assessment 3.

The assessment consists of two deep learning tasks implemented using Python, TensorFlow, and Keras.

## Task 2.1 - CNN Age and Gender Estimation

The first program uses the UTKFace dataset to develop a Convolutional Neural Network (CNN) for two related prediction tasks:

- Gender classification
- Age regression
- Comparison of training with and without data augmentation
- CNN architecture and parameter experimentation
- Evaluation using age MAE, gender accuracy, and loss curves
- Sample age and gender predictions

**Main source file:** `p1.py`

## Task 2.2 - LSTM Stock Price Forecasting

The second program uses a Long Short-Term Memory (LSTM) neural network to forecast Australian stock closing prices.

The experiments include:

- Time step = 5
- Time step = 10
- Multiple neuron configurations
- Multiple epoch configurations
- RMSE evaluation
- Training and validation loss
- Actual and predicted stock-price visualisation

**Main source file:** `p2.py`

## Project Structure

- `p1.py` - CNN age and gender estimation program
- `p2.py` - LSTM stock-price forecasting program
- `README.md` - Project documentation
- `requirements.txt` - Python dependencies
- `.gitignore` - Files excluded from version control
- `data/` - Local assessment data
- `outputs/` - Generated figures, predictions, and results
- `report/` - Report-related files

## Dataset

The UTKFace dataset is stored locally and is not committed to this repository because of its size.

Assessment stock-price datasets and generated model outputs are also excluded from version control where appropriate.

## Assessment Tasks

### Part 1 - CNN

The CNN model will use a shared convolutional feature-extraction network followed by two output heads:

- A binary classification head for gender prediction
- A regression head for age estimation

The project will compare model performance using:

- Original UTKFace images
- UTKFace images with data augmentation

Model performance will be evaluated using:

- Age Mean Absolute Error (MAE)
- Gender classification accuracy
- Age loss
- Gender loss
- Training and validation curves
- Sample predictions

### Part 2 - LSTM

The LSTM model will use historical Australian stock closing-price data to forecast future prices.

Experiments will compare:

- Time-step values of 5 and 10
- Different LSTM neuron counts
- Different training epoch counts
- At least six parameter combinations

Performance will be evaluated using Root Mean Squared Error (RMSE), training and validation loss, and comparisons between actual and predicted stock prices.

## Technologies

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- scikit-learn

Project dependencies are listed in `requirements.txt`.

## Project Status

### Task 2.1 - CNN Age and Gender Estimation

Implementation complete.

Completed features include:

- UTKFace metadata parsing and validation
- Image resizing and normalisation
- Reproducible training and validation split
- Multi-output CNN for age regression and gender classification
- Architectural and algorithmic parameter experiments
- Data augmentation
- Original-data and augmented-data CNN training
- Explicit training and validation evaluation
- Age MAE, gender accuracy, and loss curves
- CNN architecture visualisation
- Ten sample age and gender predictions per run
- Assessment-required menu:
  - Run CNN
  - Run CNN with data augmentation
  - Quit

### Task 2.2 - LSTM Stock Price Forecasting

Implementation complete.

Completed features include:

- Mining-group stock datasets: RIO, BHP, and FMG
- Chronological stock-price preprocessing
- 204-day historical training period
- 50-day forecasting/testing period
- Training-only MinMax scaling
- LSTM sequence generation for time steps 5 and 10
- Chronological training and validation split
- TensorFlow/Keras LSTM regression model
- Six controlled neuron and epoch configurations
- Parameter tuning using validation RMSE
- Selected optimal configuration for each stock and time step
- Recursive 50-day stock-price forecasting
- RMSE and MAE evaluation in Australian dollars
- Training and validation loss curves
- Actual and predicted stock-price curves
- Actual and predicted data-point visualisation
- Forecast result CSV files
- Assessment-required menu:
  - Time step = 5
  - Time step = 10
  - Quit