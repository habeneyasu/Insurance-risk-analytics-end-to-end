# Insurance Risk Analytics & Predictive Modeling

End-to-end insurance risk analytics solution for **AlphaCare Insurance Solutions (ACIS)**, focused on car insurance planning and marketing in South Africa.

## Business Objectives

- **Optimize Marketing Strategies**: Identify high-value customer segments and develop targeted marketing approaches
- **Attract New Clients**: Discover low-risk segments where premiums can be reduced to gain competitive advantage
- **Portfolio Optimization**: Analyze historical claim data to improve profitability and risk management

## Project Overview

This project analyzes 1M+ insurance records (Feb 2014 - Aug 2015) to optimize marketing strategy and identify low-risk segments for premium reduction. The solution includes exploratory data analysis, statistical hypothesis testing, and machine learning models for premium prediction.

## Key Features

- **Exploratory Data Analysis (EDA)**: Comprehensive data quality assessment, loss ratio analysis, and risk segmentation
- **Data Version Control**: DVC integration for reproducible data pipelines
- **Statistical Hypothesis Testing**: A/B testing for risk differences across provinces, zipcodes, and demographics
- **Predictive Modeling**: Linear Regression, Random Forest, and XGBoost models for claim severity, premium optimization, and claim probability prediction
- **Model Interpretability**: SHAP analysis for feature importance and business insights
- **Object-Oriented Architecture**: Modular, maintainable codebase

## Project Structure

```
├── data/              # Data files (versioned with DVC)
├── notebooks/         # Jupyter notebooks for interactive analysis
├── src/               # Source code
│   ├── data/         # Data loading and preprocessing
│   ├── analysis/     # EDA, hypothesis testing, and statistical analysis
│   └── models/       # ML model implementations and interpretability
├── reports/           # Generated reports and visualizations
└── requirements.txt   # Python dependencies
```

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/habeneyasu/Insurance-risk-analytics-end-to-end.git
cd Insurance-risk-analytics-end-to-end

# Setup environment
chmod +x setup.sh
./setup.sh

# Or manually:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Running Analyses

```bash
# Exploratory Data Analysis
python src/run_eda.py

# Hypothesis Testing
python src/run_hypothesis_tests.py

# Machine Learning Models
python src/run_ml_models.py

# Interactive exploration
jupyter notebook notebooks/01_eda_exploration.ipynb
```

### Data Management

```bash
# Pull data (if needed)
dvc pull

# Check DVC status
dvc status
```

## Key Metrics

- **Loss Ratio**: `TotalClaims / TotalPremium`
- **Claim Frequency**: Proportion of policies with at least one claim
- **Claim Severity**: Average claim amount given a claim occurred
- **Margin**: `TotalPremium - TotalClaims`
- **Dataset**: 1,000,098 records, 52 features, 23 months (Oct 2013 - Aug 2015)

## Analysis Components

### Task 1: Exploratory Data Analysis
- Data quality assessment (missing values, outliers, duplicates)
- Loss ratio analysis by province, vehicle type, and gender
- Temporal trends and geographic analysis
- Vehicle risk profiling

### Task 2: Data Version Control
- DVC setup for reproducible data pipelines
- Local remote storage configuration
- Data versioning and tracking

### Task 3: Hypothesis Testing
- Risk differences across provinces (Chi-square, ANOVA)
- Risk differences between zip codes
- Margin differences between zip codes
- Gender-based risk differences

### Task 4: Machine Learning Models
- **Claim Severity Prediction**: Regression models (Linear, Random Forest, XGBoost) to predict TotalClaims for policies with claims
- **Premium Optimization**: Regression models to predict optimal CalculatedPremiumPerTerm
- **Claim Probability Prediction**: Classification models to predict probability of claim occurrence
- **Model Interpretability**: SHAP analysis for feature importance and business insights

## Technology Stack

- **Python 3.8+**: Core language
- **Pandas/NumPy**: Data processing
- **Scikit-learn**: Machine learning algorithms
- **XGBoost**: Gradient boosting models
- **SHAP**: Model interpretability
- **Matplotlib/Seaborn**: Visualization
- **DVC**: Data version control
- **Git/GitHub**: Version control and CI/CD

## Project Status

- ✅ **Task 1**: Git, GitHub, and EDA
- ✅ **Task 2**: Data Version Control (DVC)
- ✅ **Task 3**: A/B Hypothesis Testing
- ✅ **Task 4**: Machine Learning Models

## Key Findings

- **Overall Loss Ratio**: 1.0477 (portfolio currently unprofitable)
- **High-Risk Segments**: Gauteng province (1.22), Heavy Commercial vehicles (1.63)
- **Low-Risk Opportunities**: Female drivers (0.82), Light Commercial vehicles (0.23), Bus category (0.14)
- **Statistical Insights**: Significant risk differences across provinces and zip codes; no significant gender-based differences

## Team

Facilitators: Kerod, Mahbubah, Filimon

## License

Part of the KAIM Training Portfolio.
