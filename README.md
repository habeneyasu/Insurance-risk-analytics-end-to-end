# Insurance Risk Analytics & Predictive Modeling

End-to-end insurance risk analytics solution for **AlphaCare Insurance Solutions (ACIS)**, focused on car insurance planning and marketing in South Africa.

## Overview

This project analyzes historical insurance claim data (Feb 2014 - Aug 2015) to optimize marketing strategy and identify low-risk segments for premium reduction. The solution includes exploratory data analysis, statistical hypothesis testing, and machine learning models for premium prediction.

## Key Features

- **Exploratory Data Analysis (EDA)**: Comprehensive data quality assessment, loss ratio analysis, and risk segmentation
- **Data Version Control**: DVC integration for reproducible data pipelines
- **Statistical Testing**: A/B hypothesis testing for risk differences across provinces, zipcodes, and demographics
- **Predictive Modeling**: Linear regression and ML models for premium optimization
- **Object-Oriented Architecture**: Modular, maintainable codebase

## Project Structure

```
├── data/              # Data files (versioned with DVC)
├── notebooks/         # Jupyter notebooks
├── src/               # Source code
│   ├── data/         # Data loading and preprocessing
│   ├── analysis/     # EDA and statistical analysis
│   └── models/       # ML model implementations
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

### Run EDA

```bash
# Complete EDA pipeline
python src/run_eda.py

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
- **Analysis Dimensions**: Province, VehicleType, Gender, PostalCode
- **Dataset**: 1M+ records, 52 features, 23 months

## Analysis Components

1. **Data Quality Assessment**: Missing values, outliers, duplicates
2. **Loss Ratio Analysis**: By province, vehicle type, gender
3. **Temporal Trends**: Monthly claims and premium patterns
4. **Vehicle Risk Analysis**: Make/model risk profiling
5. **Geographic Analysis**: Province and postal code risk mapping

## Technology Stack

- **Python 3.8+**: Core language
- **Pandas/NumPy**: Data processing
- **Matplotlib/Seaborn**: Visualization
- **Scikit-learn**: Machine learning
- **DVC**: Data version control
- **Git/GitHub**: Version control and CI/CD

## Project Status

- ✅ Task 1: Git, GitHub, and EDA
- ✅ Task 2: Data Version Control (DVC)
- 🔄 Task 3: A/B Hypothesis Testing
- 🔄 Task 4: Machine Learning Models

## Team

Facilitators: Kerod, Mahbubah, Filimon

## License

Part of the KAIM Training Portfolio.
