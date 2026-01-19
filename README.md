# Healthcare Demand in Australia - Data treatment and Analysis

## Project Overview

**Objective**: Model and predict the number of doctor visits using appropriate regression models.

**Dataset**: Australian Health Survey 1977-1978

**Target Variable**: `dvisits` - Number of consultations with a doctor or specialist in the past 2 weeks

## Project Structure

```
Dataset/
├── DemandHealthCareAustralia.txt          # Dataset documentation
├── HealthCareAustralia/
│   └── HealthCareAustralia.rda           # Original R data file
├── data_preparation.py                    # Main data preparation script
├── requirements.txt                       # Python dependencies
├── healthcare_cleaned.csv                 # Cleaned data generated after running data_preparation.py
└── README.md                             # This file
```

## Setup Instructions

### 1. Install Python

Make sure you have Python 3.8 or higher installed. Check your version:

```bash
python --version
```

### 2. Create and Activate Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv .venv

# Activate it:
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# On Windows (Command Prompt):
.venv\Scripts\activate.bat

# On macOS/Linux:
source .venv/bin/activate
```

### 3. Install Required Packages

```bash
pip install -r requirements.txt
```

Key packages installed:
- `pyreadr` - Read R data files
- `pandas`, `numpy` - Data manipulation
- `statsmodels` - Statistical models
- `scipy` - Statistical functions
- `matplotlib`, `seaborn` - Visualization

### 4. Run Data Preparation Pipeline

```bash
python data_preparation.py
```

This will:
- Load data from the R data file
- Perform exploratory data analysis
- Clean and transform variables
- Create new features
- Save cleaned data to `healthcare_cleaned.csv`

## Dataset Variables

### Target Variable
- **dvisits**: Number of doctor consultations in past 2 weeks

### Demographic Variables
- **sex**: Gender (1=female, 0=male)
- **age**: Age in years divided by 100
- **agesq**: Age squared
- **income**: Annual income in AUD/1000

### Health Insurance Variables
- **levyplus**: Private health insurance for private patient in public hospital (1=yes, 0=no)
- **freepor**: Government coverage - low income/recent immigrant/unemployed (1=yes, 0=no)
- **freerepa**: Government coverage - old-age/disability pension or invalid veteran (1=yes, 0=no)

### Health Status Variables
- **illness**: Number of illnesses in past 2 weeks (capped at 5)
- **actdays**: Days of reduced activity in past 2 weeks
- **hscore**: General health score (higher = worse health)
- **chcond1**: Chronic condition without activity limitation (1=yes, 0=no)
- **chcond2**: Chronic condition with activity limitation (1=yes, 0=no)

### Healthcare Utilization Variables
- **nondocco**: Consultations with non-doctor health professionals in past 2 weeks
- **hospadmi**: Hospital admissions in past 12 months (capped at 5)
- **hospdays**: Nights in hospital during most recent admission

### Medication Variables
- **medicine**: Total prescribed and nonprescribed medications in past 2 days (removed during cleaning - redundant)
- **prescrib**: Total prescribed medications in past 2 days
- **nonprescr**: Total nonprescribed medications in past 2 days

### Engineered Features (created by pipeline)
- **any_insurance**: Indicator for any insurance coverage
- **recent_hospital**: Indicator for recent hospitalization
- **medication_user**: Indicator for medication use
- **any_chronic**: Indicator for any chronic condition
- **health_status_severe**: Indicator for severe health status (chronic condition with activity limitation)

## Data Characteristics

### Target Variable (dvisits)
- **Type**: Count data
- **Sample size**: 5,190 observations
- **Actual characteristics** (from data analysis):
  - **Highly zero-inflated**: 79.79% of observations are zero (4,141 people did not visit a doctor)
  - **Overdispersed**: Variance/Mean ratio = 2.11 (variance = 0.637, mean = 0.302)
  - **Right-skewed distribution**: Max value = 9 visits
  - **Distribution**: 75% of people had 0 visits, 15% had 1 visit, 3.4% had 2 visits

### Key Data Quality Notes
1. **Historical data**: Survey from 1977-1978 (5,190 observations)
2. **No missing values**: Dataset is complete with no missing data
3. **Grouped variables**: Age and income are mid-points of ranges
4. **Censored variables**: Several variables capped at maximum values (illness ≤5, hospadmi ≤5)
5. **Variables removed**:
   - `medicine` variable (perfect collinearity: medicine = prescrib + nonpresc)
   - `constant` variable (all values = 1, no information)
6. **High correlations detected**:
   - age and agesq: 0.992 (expected, needed for quadratic effects)
   - medicine and prescrib: 0.889 (resolved by removing medicine)

## Dataset Statistics

### Overview
- **Total observations**: 5,190
- **Original variables**: 20
- **Final variables**: 23 (after removing 2 and adding 5)
- **Target variable mean**: 0.302 doctor visits per person
- **Zero-inflation rate**: 79.79%

### Variable Distribution
- **Demographics**: 52.06% female, mean age 40.6 years
- **Insurance coverage**: 44.28% private insurance, 21.02% pension coverage, 4.28% low-income coverage
- **Health conditions**: 40.31% with chronic condition (no limitation), 11.66% with chronic condition (with limitation)
- **Medication use**: 57.05% using medications (42.95% zeros)

### Count Variable Statistics
| Variable | Mean | Variance | Var/Mean | Zero % |
|----------|------|----------|----------|--------|
| dvisits | 0.302 | 0.637 | 2.11 | 79.79% |
| illness | 1.432 | 1.916 | 1.34 | 29.94% |
| actdays | 0.862 | 8.338 | 9.68 | 85.82% |
| nondocco | 0.215 | 0.932 | 4.34 | 90.87% |
| hospadmi | 0.174 | 0.258 | 1.48 | 86.53% |

## Next Steps 

1. ✓ ~~Run the data preparation script~~ 
2. Create exploratory data visualizations:
   - Distribution plots for target variable
   - Correlation heatmap (maybe)
   - Zero-inflation patterns
   - Relationship plots between predictors and target
3. Implement regression models
4. Model comparison:
   - AIC/BIC comparison
   - Cross-validation scores
   - Residual analysis
   - Prediction accuracy metrics
5. Interpret results and prepare presentation (20-25 minutes)
