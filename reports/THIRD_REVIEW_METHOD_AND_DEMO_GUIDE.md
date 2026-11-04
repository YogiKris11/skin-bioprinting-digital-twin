# Third Review — Method and Demo Guide

## Implemented workflow
1. Imported and standardised the CECT Open-Source Bioink Database.
2. Normalised pressure (kPa), printing speed (mm/s), temperature (°C) and explicitly reported nozzle diameter (µm).
3. Prepared a complete four-parameter model-fit table with 4,400 rows: 395 complete source parameter records and 4,005 scenario-derived parameter samples.
4. Trained an Isolation Forest model with median imputation, RobustScaler and 400 estimators.
5. Implemented interactive operating-point screening using the four process parameters.
6. Implemented a paginated Dataset Explorer with filters, search, visual analytics and CSV export.
7. Implemented a Parameter Lab for interactive scenario-space exploration.

## How to explain the model
“The current ML module is an unsupervised anomaly detector. It learns the multivariate distribution of the available process settings. When pressure, speed, temperature and nozzle diameter are entered, it flags whether the combination is unusual relative to the fitted reference distribution. An unusual point is not automatically defective, and a typical point is not a guarantee of print quality.”

## Data interpretation
The model-fit table contains source-linked parameter observations and parameter samples derived from source-linked process settings. The Dataset Explorer distinguishes these record bases. The scenario samples support parameter-space exploration and model prototyping; they are not independent experimental measurements.

## Not yet implemented or validated
- No supervised model predicting measured filament width or print quality: the current CECT table has no measured filament-width or quality target labels.
- No real-time connection to a bioprinter or hospital system.
- No validated skin-specific or clinical dataset.
- No clinical validation, safety certification or hospital deployment readiness.

## Demo sequence
1. **Overview:** explain the four process parameters and their distributions.
2. **Dataset Explorer:** search, filter, paginate through the full model-fit table and export CSV.
3. **ML Process Screening:** enter four parameter values and run the Isolation Forest. Interpret the result as statistical unusualness, not a quality verdict.
4. **Parameter Lab:** inspect the scenario-space distribution and export the scenario table.
5. **Methodology & Sources:** explain model configuration, record provenance and source references.

## External sources identified
See `DATASET_SOURCE_REGISTER.csv`. The Dryad resolution dataset reports measured print-diameter observations across pressure/nozzle/material setups; the Bonatti Zenodo repository contains process video frames and metadata. These external files are not included as model inputs in this application build.
