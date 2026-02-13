# COMP0047-Data-Science-Project: S&P 500 Bull/Bear Regime Classification and Prediction

## Overview

This project develops a systematic framework to:

1. Identify historical bull and bear market regimes in the S&P 500  
2. Construct regime labels using both statistical segmentation and rule-based logic  
3. Build predictive models to forecast regime transitions  
4. Provide visual and interactive analysis of regimes and predictions  

The objective is to combine market microstructure intuition, macroeconomic context, and modern time-series modelling techniques into a reproducible research pipeline.

---

## Research Objective

To classify and predict bull/bear regimes in the S&P 500 using:

- Technical indicators  
  - Momentum  
  - MACD  
  - RSI  
  - Volatility (VIX)  
  - Volume (SPY)  

- Macroeconomic indicators  
  - Real GDP  
  - Core Inflation  
  - Unemployment Rate  
  - Money Supply  

The project integrates change-point detection, smoothing methods, deep learning models, and Bayesian techniques for regime forecasting.

---

## Data Sources

### Market Data
- Yahoo Finance  
  - `^GSPC` (S&P 500 Index)  
  - `VIX` (Volatility Index)  
  - `SPY` (Volume proxy)  

### Macroeconomic Data
- FRED (Federal Reserve Economic Data)  
  - `GDPC1` — Real GDP  
  - `PCEPILFE` — Core PCE Inflation  
  - `UNRATE` — Unemployment Rate  
  - `M2SL` — Money Supply  

### Optional
- Refinitiv sentiment indicators  

---

## Methodology

### 1. Data Engineering
- Download and align market and macro time series
- Frequency harmonisation (daily vs monthly)
- Interpolation and forward-filling where appropriate
- Feature normalization and scaling

### 2. Feature Engineering
- Technical indicators (momentum, MACD, RSI)
- Time-lagged features
- Rolling statistics and multiscale aggregation
- Smoothing techniques (Savitzky–Golay filter, etc.)

### 3. Regime Segmentation
Unsupervised regime discovery methods:
- ClaSP
- RBEAST
- Change-point detection

### 4. Label Construction
- Rule-based bull/bear classification
- Volatility threshold rules
- Composite regime scoring

### 5. Predictive Modelling
- RNN
- LSTM
- Bayesian models
- Transformer-based time-series models (TST / TimeSeriesTransformer)

### 6. Evaluation
- Classification accuracy
- Precision / Recall / F1
- Regime transition accuracy
- Confusion matrix analysis
- Optional backtesting of regime-aware strategies

---

## Repository Structure

```
data/
  raw/
  interim/
  processed/

notebooks/
  Exploratory analysis and modelling experiments

src/
  data/
  features/
  models/
  utils/

reports/
  Final research report and figures

dashboard/
  Interactive Plotly regime visualisation

experiments/
  Model logs and saved weights
```

---

## Installation

### Using pip

```bash
pip install -r requirements.txt
```

### Using conda

```bash
conda env create -f environment.yaml
conda activate COMP0047-Data-Science-Project
```

---

## Reproducibility Workflow

1. Run data ingestion scripts in `src/data`
2. Generate features using `src/features`
3. Create regime labels
4. Train predictive models
5. Evaluate and visualise results
6. Generate report figures

---

## Outputs

- Regime-labelled dataset
- Predictive modelling notebook
- Interactive Plotly visualisation
- Research report (PDF)
- Saved trained models

---

## License

MIT License

---