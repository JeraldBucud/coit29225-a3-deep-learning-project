# COIT29225 Assessment 3 — Neural Networks and Deep Learning

**CQUniversity | Term 2, 2026**

This repository contains the Python implementations for two Assessment 3 tasks:

1. **Part 1 — CNN age and gender estimation** using the UTKFace facial-image dataset.
2. **Part 2 — LSTM stock-price forecasting** using Australian mining-company closing-price data (RIO, BHP and FMG).

Both applications use TensorFlow/Keras, provide the required three-option terminal menu, and save experiment evidence into separate subdirectories under `outputs/`.

## Project layout

```text
Assessment 3/
├── p1.py                  # CNN: age regression and gender classification
├── p2.py                  # LSTM: stock-closing-price forecasting
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── .gitkeep
│   ├── RIO.csv              # Local input, ignored by Git
│   ├── BHP.csv
│   └── FMG.csv
├── outputs/
│   ├── .gitkeep            # Keeps the parent directory in Git
│   ├── p1/                 # Generated CNN outputs (ignored by Git)
│   │   ├── cnn_original_*.png / .csv / .keras
│   │   ├── cnn_augmented_*.png / .csv / .keras
│   │   ├── cnn_final_comparison.csv
│   │   └── tuning/
│   │       ├── cnn_tuning_manifest.json
│   │       ├── cnn_experiment_results.csv
│   │       └── cnn_experiment_*_history.csv
│   └── p2/                 # Generated LSTM outputs (ignored by Git)
│       ├── lstm_parameter_experiment_results.csv
│       ├── lstm_t5_final_results.csv
│       ├── lstm_t10_final_results.csv
│       ├── lstm_<STOCK>_t<STEP>_forecast.csv / .png
│       ├── lstm_<STOCK>_t<STEP>_loss.png
│       ├── lstm_<STOCK>_t<STEP>.keras
│       └── tuning/
│           ├── lstm_tuning_manifest.json
│           └── histories/  # 36 per-experiment training-history CSVs
└── report/                # Local report materials
```

The `outputs/p1/` and `outputs/p2/` directories are created automatically when the programs run. Generated models and dataset files are deliberately not tracked by Git.

## Setup

The project was developed with **Python 3.11** in the Conda environment `coit29225-a3`. The recorded training environment used TensorFlow **2.21.0**. Package versions are specified in `requirements.txt`.

In **Anaconda Prompt** or a PowerShell session configured for Conda:

```powershell
conda activate coit29225-a3
python -m pip install -r requirements.txt
```

Verify the active interpreter and key libraries:

```powershell
python -c "import sys, tensorflow as tf, matplotlib; print(sys.executable); print('TensorFlow:', tf.__version__); print('Matplotlib OK')"
```

**Input data:**

- **P1:** `p1.py` expects UTKFace image files in `~/Desktop/UTKFace`. Adjust `DATASET_PATH` in `p1.py` if your local dataset is elsewhere. The entire dataset is excluded from Git.
- **P2:** Place the supplied `RIO.csv`, `BHP.csv`, and `FMG.csv` files in `data/`. Each expected file contains 254 trading-day records, including `Date` and `Close $` columns.

## Part 1 — CNN age and gender estimation (`p1.py`)

The CNN uses a shared convolutional feature-extraction backbone with two heads:

- **Age:** single linear output; mean squared error (MSE) training loss; mean absolute error (MAE) evaluation in years.
- **Gender:** single sigmoid output; binary cross-entropy training loss; classification accuracy evaluation.

Images are decoded as RGB, resized to **128 × 128**, and normalised to **0–1**. The training/validation split is **80/20**. The augmentation variant applies random horizontal flips, rotation, zoom, and contrast adjustments during training.

### Run the required menu

```powershell
python p1.py
```

```text
1. Run CNN
2. Run CNN with data augmentation
3. Quit
```

### Reproduce the four parameter experiments

```powershell
python p1.py --tune
```

The tuning workflow compares baseline, compact, wider, and regularised CNN configurations. It saves a results table, an individual epoch history for each configuration, and a reproducibility manifest in `outputs/p1/tuning/`.

Optional: `python p1.py --tune --save-tuning-models` additionally saves the four large tuning model files. To regenerate the saved final prediction plots from their existing data without retraining, run `python p1.py --refresh-prediction-plots`.

### Recorded final P1 results

| Variant | Validation age MAE (years) ↓ | Validation gender accuracy ↑ |
|---|---:|---:|
| Original | 7.0409 | 89.35% |
| Augmented | **6.8710** | 88.76% |

Source: `outputs/p1/cnn_final_comparison.csv` from the verified final run. Augmentation marginally improves the age metric, but the original model has slightly higher gender accuracy.

## Part 2 — LSTM stock-price forecasting (`p2.py`)

The assigned mining group consists of **Rio Tinto (RIO), BHP (BHP), and Fortescue (FMG)**.

For each stock, the chronological **254-record** dataset is split into **204 historical training observations** and a **50-day held-out forecast/test period**. The LSTM uses a **5- or 10-day** lookback window, followed by a single linear Dense output. Final forecasting is **recursive**: predicted values are reused to build later inputs. No actual held-out future prices are supplied to the forecast loop.

The parameter-tuning stage uses an internal chronological validation split and fits its MinMax scaler only on tuning-training prices, avoiding validation/future-target leakage. The final selected model is trained on the 204 historical observations; the held-out 50 days are used only for retrospective evaluation.

### Run the required menu

```powershell
python p2.py
```

```text
1. Time step = 5
2. Time step = 10
3. Quit
```

The first menu run may initiate tuning if compatible cached tuning evidence is not available. Selecting `1` or `2` trains and evaluates the three stock-specific final models for that time step.

### Reproduce the 36 tuning experiments

```powershell
python p2.py --tune
```

Six configurations (16/32/64 LSTM units × 20/50 epochs) are tested for each stock and each time step: **3 × 2 × 6 = 36 experiments**. This saves `outputs/p2/lstm_parameter_experiment_results.csv`, all 36 per-experiment histories in `outputs/p2/tuning/histories/`, and `outputs/p2/tuning/lstm_tuning_manifest.json`.

The manifest records the source/data fingerprints, experiment configuration and execution environment. Compatible partial experiments can be resumed; stale cached evidence is not accepted for final model selection.

### Recorded final P2 results

| Stock | Time-step-5 forecast RMSE (AUD) ↓ | Time-step-10 forecast RMSE (AUD) ↓ | Last-price baseline RMSE (AUD) ↓ |
|---|---:|---:|---:|
| RIO | 7.3463 | **6.4662** | 9.4981 |
| BHP | **3.3441** | 3.7615 | 5.4177 |
| FMG | 1.9133 | **1.5686** | **0.8320** |

Sources: `outputs/p2/lstm_t5_final_results.csv` and `outputs/p2/lstm_t10_final_results.csv` from the verified final run. The best LSTM outperforms the last-price baseline for RIO and BHP, but **not for FMG**. Final forecast RMSE must not be confused with the separate tuning-validation RMSE used to select configurations.

## Reproducibility and evidence

- Both programs use **random seed 42** for repeatability, although TensorFlow results can vary slightly across software versions and hardware.
- P1 tuning records model settings, experiment histories and a manifest; P2 saves an analogous manifest and 36 experiment histories.
- PNG plots, prediction CSVs, run summaries, and trained `.keras` models are generated locally under the appropriate output folder.
- Re-running training can overwrite outputs with the same filename. Back up evidence used in the final report before restarting experiments.
- **Only commit code and lightweight documentation to Git by default.** The `.gitignore` intentionally excludes raw datasets and generated model/figure artifacts. Keep the submitted report and video evidence separately, as required by the assessment instructions.

## Assessment deliverables

Use `p1.py`, `p2.py`, the final **combined P1 + P2 report**, and the separately recorded **4–6-minute presentation video** to prepare the required submission. This repository README is supplementary documentation and does not replace the assessed report.
