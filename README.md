# IEPFA Empirical Analysis

Empirical analysis and time series forecasting of unclaimed financial assets under the Investor Education and Protection Fund Authority (IEPFA) in India.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Run the main analysis pipeline:

```bash
python scripts/empirical_analysis.py
```

This estimates the regression models, produces the event study and awareness campaign figures, and runs the ARIMA forecasting models.

## Project Structure

- `data/`: Excel data files
- `scripts/`: Python scripts for empirical analysis and ARIMA models
- `plots/`: Output figures, regression results, and forecast summaries
- `paper/`: LaTeX report files and figures

