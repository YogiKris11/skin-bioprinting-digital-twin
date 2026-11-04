# Skin Bioprinting Digital Twin

## Launch on Windows
1. Extract the ZIP archive.
2. Open the extracted `Skin_Bioprinting_Digital_Twin_Professional` folder.
3. Double-click `RUN_DEMO_NOW.bat`.
4. The launcher installs the Python dependencies and starts the Streamlit dashboard.

## Application modules
- **Overview:** process parameter distributions, record composition and multivariate process-space analytics.
- **ML Process Screening:** interactive four-parameter Isolation Forest anomaly screening.
- **Dataset Explorer:** searchable, filterable, paginated view of the complete 4,400-row model-fit parameter table, with CSV exports.
- **Parameter Lab:** interactive exploration and export of the 5,000-row parameter-space scenario table.
- **Methodology & Sources:** model interpretation, data basis and source references.

## Model interpretation
The implemented model is an unsupervised Isolation Forest. It flags operating points that are statistically unusual relative to its fitted parameter distribution. It is not a supervised print-quality classifier. The model-fit table contains complete four-parameter source-linked records and scenario-derived parameter samples; the explorer identifies these record bases separately. The current CECT source table does not provide measured filament-width or print-quality target labels.

## Requirements
Python 3.10 or newer is recommended. Internet access is required on first launch so the launcher can install the packages listed in `requirements.txt`.
