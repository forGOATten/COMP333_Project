# COMP 333 — Final Project: End-to-End Data Analytics Pipeline

## Team Information

| Name | Student ID | Contribution |
|------|-----------|--------------|
| Gorden | 40263250 | - readme, github repo, Baseline Model Execution |
| [Teammate 2] | [ID] | [Describe contributions] |

## Project Overview

This project implements a complete data analytics pipeline on the **Bixi 2025 Trip History** dataset, enriched with **Environment Canada Hourly Climate** data. The goal is to investigate how environmental conditions influence urban cycling behavior in Montreal.

### Research Questions

1. **Supervised: Predictive Regression** To what degree can temperature and precipitation predict Bixi trip duration?
2. **Unsupervised: Behavioral Clustering** 2. **Unsupervised: Behavioral Clustering** By clustering stations based on their 'Net Flow' (Returns vs Departures)


## Dataset
### 1. 
- **Name:** Bixi 2025 Trip History
- **Source:** https://bixi.com/en/open-data/
- **Size:** 2.60GB
- **Records:** 14.2m
- **Features:** 10 columns

### 2. 
- **Name:** Environment Canada Weather (Hourly Climate)
- **Source:**https://climate-change.canada.ca/climate-data/#/hourly-climate-data
- **Size:**  4.6 MB (Combined Jan-Dec)
- **Records:** 17,977
- **Features:** 40 columns

### Download Instructions
1. **Bixi data:** Download the 2025 trip history CSV from [bixi.com/en/open-data/](https://bixi.com/en/open-data/) and place it in `data/raw/Bixi2025.csv`
2. **Weather data:** Download hourly climate data for Montreal (McTavish station) from [Climate Change Canada](https://climate-change.canada.ca/climate-data/#/hourly-climate-data). Save as two files in `data/raw/`:
   - `HourlyClimate - JanToMar.csv`
   - `HourlyClimate - MarToDec.csv`

## Setup & Reproduction

1. Clone the repository:
   ```bash
   git clone https://github.com/forGOATten/COMP333_Project
   cd COMP333_Project
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
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
pandas, numpy, matplotlib, seaborn, scikit-learn, scipy

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
- BIXI Montréal. "Open Data." BIXI Montréal, https://bixi.com/en/open-data/
- Government of Canada. "Hourly Climate Data." Climate Change Canada, https://climate-change.canada.ca/climate-data/#/hourly-climate-data
- Société de transport de Montréal (STM). "Developers." STM, https://www.stm.info/en/about/developers
- Breiman, L., Friedman, J., Stone, C. J., & Olshen, R. A. (1984). *Classification and Regression Trees*. CRC Press.
- Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.). O'Reilly Media.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.
- Pedregosa, F., et al. "Scikit-learn: Machine Learning in Python." *Journal of Machine Learning Research*, vol. 12, 2011, pp. 2825–2830. https://scikit-learn.org/stable/
- pandas Development Team. "pandas: Powerful Python Data Analysis Toolkit." https://pandas.pydata.org/docs/
- Hunter, J.D. "Matplotlib: A 2D Graphics Environment." *Computing in Science & Engineering*, vol. 9, no. 3, 2007, pp. 90–95. https://matplotlib.org/