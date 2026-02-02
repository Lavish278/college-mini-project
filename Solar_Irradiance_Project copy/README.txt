AI-BASED SOLAR IRRADIANCE FORECASTING (MATLAB PROJECT)

========================================================
PROJECT DESCRIPTION
========================================================

This project presents a complete and reproducible MATLAB implementation
for forecasting Global Horizontal Irradiance (GHI) using machine learning
and deep learning techniques.

The objective is to compare traditional regression, ensemble learning,
artificial neural networks, and sequence-based deep learning models
for short-term solar irradiance prediction.

The project is designed for:
- Renewable energy forecasting
- Solar power planning
- Academic research and thesis work
- MATLAB-based AI experimentation


========================================================
PROJECT STRUCTURE
========================================================

Root Directory
|
|-- data
|   |-- raw
|       |-- Solar_Irradiance_2017_15min.csv
|       |-- Solar_Irradiance_2017_30min.csv
|       |-- Solar_Irradiance_2017_60min.csv
|       |-- Solar_Irradiance_2018_15min.csv
|       |-- Solar_Irradiance_2018_30min.csv
|       |-- Solar_Irradiance_2018_60min.csv
|       |-- Solar_Irradiance_2019_15min.csv
|       |-- Solar_Irradiance_2019_30min.csv
|       |-- Solar_Irradiance_2019_60min.csv
|
|-- results
|   |-- figures
|       |-- Actual_vs_LR.png
|       |-- Actual_vs_SVR.png
|       |-- Actual_vs_RF.png
|       |-- Actual_vs_ANN.png
|       |-- Actual_vs_LSTM.png
|   |
|   |-- tables
|       |-- Final_Model_Results_Extended.mat
|       |-- Final_Model_Results_Extended.xlsx
|   |
|   |-- predictions
|       |-- Final_Model_Predictions.mat
|
|-- logs
|   |-- Solar_Irradiance_CommandLog.txt
|
|-- init_project_structure.m
|-- Solar_Irradiance_Forecasting_Final.m
|-- README.txt


========================================================
DATASET DESCRIPTION
========================================================

Years Covered:
- 2017
- 2018
- 2019

Time Resolutions:
- 15 minutes
- 30 minutes
- 60 minutes

Target Variable:
- Global Horizontal Irradiance (GHI)

Input Features Include:
- Clearsky GHI
- Clearsky DNI
- Clearsky DHI
- DNI
- DHI
- Temperature
- Relative Humidity
- Dew Point
- Pressure
- Wind Speed
- Wind Direction
- Precipitable Water
- Cloud Type

Night-time samples (GHI <= 0) are automatically removed.


========================================================
MODELS IMPLEMENTED
========================================================

The following models are trained and evaluated:

1. Linear Regression (LR)
2. Support Vector Regression (SVR) with RBF kernel
3. Random Forest Regression (RF)
4. Artificial Neural Network (ANN)
5. Long Short-Term Memory Network (LSTM)

All models use:
- Same dataset
- Same preprocessing
- Same 80% training / 20% testing split
- Chronological (time-respecting) split


========================================================
EVALUATION METRICS
========================================================

The following performance metrics are computed for each model:

RMSE   : Root Mean Squared Error
MAE    : Mean Absolute Error
MAPE   : Mean Absolute Percentage Error
nRMSE  : Normalized RMSE
MBE    : Mean Bias Error
R2     : Coefficient of Determination
PearsonR : Pearson Correlation Coefficient
NSE    : Nash–Sutcliffe Efficiency

These metrics jointly evaluate accuracy, bias, stability,
and correlation between predicted and observed GHI values.


========================================================
SUMMARY OF RESULTS (TEST SET)
========================================================

Model   RMSE      MAE       R2
--------------------------------
LR      0.1657    0.1161    0.9718
SVR     0.0855    0.0594    0.9925
RF      0.0306    0.0171    0.9990
ANN     0.0015    0.0009    1.0000
LSTM    0.0116    0.0082    0.9999

Key Observations:
- ANN achieves the lowest overall error.
- LSTM captures temporal dependencies effectively.
- Random Forest performs strongly among non-deep models.
- Linear Regression performs weakest due to nonlinearity.


========================================================
HOW TO RUN THE PROJECT
========================================================

Step 1: Open MATLAB and set the project root folder
        as the current working directory.

Step 2: Initialize folder structure:
        >> init_project_structure

Step 3: Run the full forecasting pipeline:
        >> Solar_Irradiance_Forecasting_Final

The script will automatically:
- Load all datasets
- Preprocess and normalize data
- Train all models with progress visualization
- Compute performance metrics
- Save figures, tables, predictions, and logs


========================================================
OUTPUT FILES GENERATED
========================================================

Figures:
- results/figures/Actual_vs_LR.png
- results/figures/Actual_vs_SVR.png
- results/figures/Actual_vs_RF.png
- results/figures/Actual_vs_ANN.png
- results/figures/Actual_vs_LSTM.png

Tables:
- results/tables/Final_Model_Results_Extended.xlsx
- results/tables/Final_Model_Results_Extended.mat

Predictions:
- results/predictions/Final_Model_Predictions.mat

Execution Log:
- logs/Solar_Irradiance_CommandLog.txt


========================================================
REPRODUCIBILITY FEATURES
========================================================

- Automatic directory creation
- Command window logging
- Fixed train/test split
- No manual intervention
- Fully script-driven execution


========================================================
INTENDED USE
========================================================

This project is suitable for:
- Academic research papers
- Master’s and PhD theses
- Renewable energy forecasting studies
- MATLAB machine learning demonstrations


========================================================
PROJECT STATUS
========================================================

Code finalized
Results validated
Figures generated
Tables exported
Repository ready for GitHub submission
