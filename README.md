# COMP 333 — Final Project: End-to-End Data Analytics Pipeline

## Team Information

| Name | Student ID | Contribution |
|------|-----------|--------------|
| Gorden | [Your ID] | [Describe contributions] |
| [Teammate 2] | [ID] | [Describe contributions] |

## Project Overview

This project implements a complete data analytics pipeline on the **US Accidents (2016–2023)** dataset, exploring crash severity prediction based on weather and road conditions.

### Research Questions

1. **Supervised:** Can we predict the severity of a traffic accident (binary classification) based on weather conditions, road features, and time-of-day factors?
2. **Unsupervised:** Are there natural clusters of accident types based on environmental and road conditions?

## Dataset

- **Name:** US Accidents (2016–2023)
- **Source:** [Kaggle — US Accidents](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)
- **Size:** ~3 GB (uncompressed)
- **Records:** ~7.7 million accident records
- **Features:** 46 columns including weather, location, road conditions, and severity

### Download Instructions

1. Install the Kaggle CLI (if not already installed):
   ```bash
   pip install kaggle
   ```

2. Set up your Kaggle API credentials:
   - Go to [kaggle.com/settings](https://www.kaggle.com/settings) → Create New Token
   - Place `kaggle.json` in `~/.kaggle/`

3. Download the dataset:
   ```bash
   kaggle datasets download -d sobhanmoosavi/us-accidents -p data/
   unzip data/us-accidents.zip -d data/
   ```

4. Verify the file is located at:
   ```
   data/US_Accidents_March23.csv
   ```

> **Note:** The dataset is too large for Git. Do NOT commit the CSV file — it is included in `.gitignore`.

## Project Structure

```
├── README.md
├── requirements.txt
├── .gitignore
├── data/                     # Dataset goes here (not tracked by Git)
├── notebooks/
│   ├── Phase1_EDA_Baseline.ipynb
│   ├── Phase2_Advanced_Modeling.ipynb
│   └── Phase3_Full_Pipeline.ipynb
├── exports/                  # PDF exports of notebooks
│   ├── Phase1_EDA_Baseline.pdf
│   ├── Phase2_Advanced_Modeling.pdf
│   └── Phase3_Full_Pipeline.pdf
└── src/                      # Optional: refactored modules for Phase 3
    └── __init__.py
```

## Setup & Reproduction

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd <repo-name>
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the dataset (see instructions above).

5. Open and run the notebooks in order:
   ```bash
   jupyter notebook
   ```

## Dependencies

See `requirements.txt` for the full list. Key libraries include:
pandas, numpy, matplotlib, seaborn, scikit-learn, scipy, xgboost, lightgbm

## Phases & Deadlines

| Phase | Description | Due Date |
|-------|-------------|----------|
| Phase 1 | Data Acquisition & Baseline | March 1, 2026 |
| Phase 2 | Advanced Modeling | April 5, 2026 |
| Phase 3 | Complete Pipeline & Demo | Final Exam |

## AI Tool Usage

This project uses AI coding assistants (Claude) for boilerplate code generation, debugging assistance, and documentation. All code is reviewed and understood by team members.

## References

- Moosavi, Sobhan, et al. "A Countrywide Traffic Accident Dataset." (2019).
- [Kaggle — US Accidents Dataset](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)