# Rainfall Forecasting Repository

This repository contains the implementation for a rainfall forecasting project using various meteorological and oceanographic parameters in Malaysia region specifically. The project leverages historical weather data to predict daily rainfall amounts.

## Project Overview
Rainfall forecasting is crucial for improving agricultural planning, flood prevention, and urban planning in Malaysia. By accurately predicting daily rainfall amounts, this project aims to assist government agencies, local communities, and industries in making informed decisions to mitigate risks and optimize resource management.

## Dataset Description
The dataset used for this project includes the following features:

- **DateTime**: Date and time of the recorded data (YYYY-MM-DD HH:MM:SS) in daily intervals.
- **Rainfall**: Total rainfall amount (millimeters) recorded during the specified time period.
- **TOTAL**: Observed sea surface temperature (SST) in the Niño 3.4 region.
- **ClimAdjust**: Climatologically adjusted SST, accounting for long-term trends and seasonal variations.
- **ANOM**: Anomaly value, representing the difference between the observed SST and the climatologically adjusted SST.
- **Temperature**: Ambient temperature (Fahrenheit) at the time of measurement.
- **DewPoint**: Temperature at which air becomes saturated, indicating moisture content.
- **Humidity**: Relative humidity or moisture in the air (percentage).
- **Visibility**: Distance (miles) over which objects can be observed clearly.
- **WindSpeed**: Wind speed (meters per hour) at the time of observation.
- **Pressure**: Atmospheric pressure (inches) at the recording location.
- **Latitude**: Geographical latitude of the recording station.
- **Longitude**: Geographical longitude of the recording station.
- **Elevation**: Elevation of the recording station, affecting weather patterns.
- **Wind**: Degree of wind direction at the time of measurement.
- **Condition**: General description of the weather conditions during the recording.
<br /><br />
> **Disclaimer:** The data used in this project is sourced from publicly available repositories and meteorological databases. Specifically, data has been referenced from the [Oceanic Niño index (ONI)](https://origin.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_change.shtml), [Weather Underground](https://www.wunderground.com/), [Jabatan Meteorologi Malaysia](https://www.met.gov.my/), [iHydro DID Sarawak](https://ihydro.sarawak.gov.my/iHydro/en/index.jsp), and [topographic-map.com](https://en-gb.topographic-map.com/). These sources provide data with varying granularity, coverage areas, and temporal resolutions, which are critical for reliable forecasting. The accuracy and reliability of the predictions depend on the quality of the input data.

## Key Features
- **Data Preprocessing**: Handles missing values, outliers, and data transformations.
- **Exploratory Data Analysis (EDA)**: Visualizes data distributions, correlations, and trends.
- **Model Training**: Implements machine learning  and deep learning models for rainfall forecasting, including feature engineering and hyperparameter tuning.
- **Evaluation Metrics**: Uses metrics such as Mean Absolute Error (MAE) and R² Score for performance evaluation.
- **Visualization**: Provides detailed plots for insights into model performance and data characteristics.

## Model Details
This project employs a combination of machine learning and deep learning models such as Random Forest and LSTM for rainfall forecasting. The Random Forest model captures non-linear relationships between features, while the LSTM model is optimized for capturing temporal patterns in the data. Hyperparameter tuning is applied to improve model performance, and cross-validation ensures generalizability.

## Future Work
This project aims to find the best model suited for forecasting the daily amount of rainfall in the Malaysia region, taking into account ONI data from ENSO.