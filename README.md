# COMP 333 — Final Project: End-to-End Data Analytics Pipeline

## Team Information

| Name | Student ID | Contribution |
|------|-----------|--------------|
| Gorden | 40263250 | - readme, github repo,  |
| [Teammate 2] | [ID] | [Describe contributions] |

## Project Overview

This project implements a complete data analytics pipeline on the **Bixi 2025 Trip History, Environment Canada Weather (Hourly Climate), STM (Metro station locations)** dataset.

### Research Questions

1. **Supervised:** Can we predict trip duration or categorize trips as "Long vs. Short" by integrating time-of-day, real-time weather conditions, and proximity to the nearest Metro?
2. **Unsupervised:** By clustering stations based on their 'Net Flow' (Returns vs Departures)
and their distance to the nearest Metro, can we programmatically identify the city’s Commuter vs Leisure transit zones?

## Dataset
### 1. 
- **Name:** Bixi 2025 Trip History
- **Source:** https://bixi.com/en/open-data/
- **Size:** 
- **Records:** 
- **Features:** 

### 2. 
- **Name:** Environment Canada Weather (Hourly Climate)
- **Source:**https://climate-change.canada.ca/climate-data/#/hourly-climate-data
- **Size:** 
- **Records:** 
- **Features:** 

### 3. 
- **Name:** STM (Metro station locations)
- **Source:** https://www.stm.info/en/about/developers
- **Size:** 
- **Records:** 
- **Features:** 

### Download Instructions
[to complete]

## Setup & Reproduction

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd <repo-name>
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: venv\Scripts\activate
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
(update as we go)
See `requirements.txt` for the full list. Key libraries include:
pandas, numpy, matplotlib, seaborn, scikit-learn, scipy, xgboost, lightgbm

## Phases & Deadlines

| Phase | Description | Due Date |
|-------|-------------|----------|
| Phase 1 | Data Acquisition & Baseline | March 1, 2026 |
| Phase 2 | Advanced Modeling | April 5, 2026 |
| Phase 3 | Complete Pipeline & Demo | Final Exam |

### Phase 1: Data Acquisition & Baseline

- **Data Retrieval:** Programmatic retrieval of the 2GB+ Bixi CSV and retrieval of STM/Weather coordinates.
- **Wrangling:** Implement a reproducible pipeline to convert Unix timestamps and filter outliers.
- **Baseline Model:** Train a Simple Linear/Logistic Regression (70/15/15 split) to establish a performance floor.
- **EDA:** Use `quantDDA()` and `vizDDA()` to visualize trip density and weather correlations.

### Phase 2: Advanced Modeling & Feature Engineering

- **Advanced Supervised Learning:** Implement appropriate models to capture non-linear weather relationships.
- **Feature Engineering:** Create domain-specific features (e.g., `Is_Rush_Hour`, `Is_Weekend`) and interaction features.
- **Unsupervised:** Implement dimension reduction and determine optimal station clusters. Visualize; evaluate quality; justify appropriateness.

### Phase 3: Pipeline & Ethics

- **End-to-End Pipeline:** Refactor logic into modular, error-handled Python functions for reproducibility.
- **Ethical Considerations (Model Fairness):** We will analyze "Majority Bias" within our model. Because 90% of data represents fair-weather daytime trips, we will evaluate if our model "marginalizes" night-shift or all-weather commuters by being less accurate for minority conditions.

## AI Tool Usage

This project uses AI coding assistants for boilerplate code generation, debugging assistance, and documentation. All code is reviewed and understood by team members.

## References
(update as we go)