# Electricity Load Forecasting under Extreme Weather Conditions

## Project Overview

This project focuses on short-term electricity load forecasting under normal and extreme weather conditions.

Electricity demand is strongly influenced by historical consumption patterns and meteorological factors. However, extreme weather events introduce additional uncertainties and distribution shifts, making accurate forecasting more challenging.

This project investigates how different time-series forecasting architectures adapt to weather-driven load variations, with a focus on model performance, robustness, and interpretability.

The main objectives are:

- Construct a weather-integrated electricity load forecasting dataset
- Compare different forecasting architectures under the same experimental setting
- Analyze model robustness under extreme weather scenarios
- Interpret model behavior using Explainable AI (XAI)

---

## Research Question

How do different time-series forecasting models capture the relationship between historical electricity demand and weather disturbances, especially under extreme weather conditions?

---

## Methods

The project compares several forecasting approaches:

- **DLinear**  
  A lightweight linear forecasting model based on trend and seasonal decomposition.

- **Autoformer**  
  A Transformer-based time-series forecasting model using series decomposition and autocorrelation mechanisms.

- **iTransformer**  
  A Transformer variant designed for multivariate time-series forecasting by modeling relationships among variables.

- **Random Forest**  
  A traditional machine learning baseline.

The models are evaluated on both normal weather and extreme weather scenarios.

---

## Dataset

The dataset combines:

- Belgium electricity load data from Elia
- Weather variables from ECMWF

Meteorological features include:

- Temperature
- Wind speed
- Sea level pressure
- Precipitation
- Cloud cover
- Solar radiation

The forecasting task is formulated as:

- Input: historical load and weather variables
- Output: future electricity load

Experimental setting:

- Input length: 168 hours (7 days)
- Forecast horizon: 24 hours
- Task type: Multivariate input, single target forecasting

---

## Key Contributions

This project focuses on experimental analysis rather than proposing a new forecasting architecture.

The main contributions include:

1. Building a weather-aware electricity load forecasting pipeline.
2. Evaluating multiple forecasting architectures under normal and extreme weather conditions.
3. Analyzing the influence of weather factors and model behaviors using SHAP-based interpretation.
4. Investigating the relationship between model structure and forecasting performance.

---

## Results Summary

iTransformer achieved the best overall performance among evaluated models in both normal and extreme weather scenarios.

DLinear also demonstrated competitive performance despite its simple structure, showing that effective forecasting does not always require highly complex architectures.

SHAP analysis was further applied to investigate how different models utilize historical load information and weather variables.