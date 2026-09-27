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

Additional dependencies will be added as the project develops.

## Project Status

Initial project setup.