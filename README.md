# Insurance Risk Analytics & Predictive Modeling

## Project Overview

This project is an end-to-end insurance risk analytics solution for **AlphaCare Insurance Solutions (ACIS)**, focused on car insurance planning and marketing in South Africa. The objective is to analyze historical insurance claim data to optimize marketing strategy and discover "low-risk" targets for premium reduction, thereby attracting new clients.

## Business Objective

Develop cutting-edge risk and predictive analytics to:
- Optimize marketing strategy
- Discover low-risk segments for premium reduction
- Build predictive models for optimal premium pricing
- Provide actionable insights for product tailoring

## Project Structure

```
Insurance-risk-analytics-end-to-end/
├── data/                      # Data files (versioned with DVC)
├── notebooks/                 # Jupyter notebooks for exploration
├── src/                       # Source code
│   ├── data/                 # Data processing modules
│   ├── analysis/             # Statistical analysis modules
│   ├── models/               # ML model implementations
│   └── utils/                # Utility functions
├── tests/                     # Unit tests
├── reports/                   # Generated reports and visualizations
├── .github/                   # GitHub Actions workflows
│   └── workflows/
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore rules
└── README.md                # This file
```

## Key Analysis Areas

### 1. Insurance Terminologies
Understanding key insurance concepts and terminology.

### 2. A/B Hypothesis Testing
Testing the following null hypotheses:
- No risk differences across provinces
- No risk differences between zipcodes
- No significant margin (profit) difference between zip codes
- No significant risk difference between Women and men

### 3. Machine Learning & Statistical Modeling
- Linear regression models per zipcode to predict total claims
- ML model for optimal premium prediction based on:
  - Car features
  - Owner features
  - Location features
  - Other relevant features
- Feature importance analysis

## Data Description

**Time Period:** February 2014 to August 2015

### Data Columns

#### Insurance Policy
- `UnderwrittenCoverID`
- `PolicyID`
- `TransactionMonth`

#### Client Information
- `IsVATRegistered`
- `Citizenship`
- `LegalType`
- `Title`
- `Language`
- `Bank`
- `AccountType`
- `MaritalStatus`
- `Gender`

#### Location
- `Country`
- `Province`
- `PostalCode`
- `MainCrestaZone`
- `SubCrestaZone`

#### Vehicle Information
- `ItemType`
- `Mmcode`
- `VehicleType`
- `RegistrationYear`
- `Make`
- `Model`
- `Cylinders`
- `Cubiccapacity`
- `Kilowatts`
- `Bodytype`
- `NumberOfDoors`
- `VehicleIntroDate`
- `CustomValueEstimate`
- `AlarmImmobiliser`
- `TrackingDevice`
- `CapitalOutstanding`
- `NewVehicle`
- `WrittenOff`
- `Rebuilt`
- `Converted`
- `CrossBorder`
- `NumberOfVehiclesInFleet`

#### Plan Details
- `SumInsured`
- `TermFrequency`
- `CalculatedPremiumPerTerm`
- `ExcessSelected`
- `CoverCategory`
- `CoverType`
- `CoverGroup`
- `Section`
- `Product`
- `StatutoryClass`
- `StatutoryRiskType`

#### Financial Metrics
- `TotalPremium`
- `TotalClaims`

## Tasks

### Task 1: Git, GitHub, and EDA
- [x] Git repository setup
- [ ] GitHub Actions CI/CD
- [ ] Exploratory Data Analysis (EDA)
- [ ] Statistical analysis and visualizations

### Task 2: Data Version Control (DVC)
- [ ] DVC setup and configuration
- [ ] Data versioning
- [ ] Local remote storage setup

### Task 3: Hypothesis Testing
- [ ] A/B testing implementation
- [ ] Statistical hypothesis tests
- [ ] Results interpretation

### Task 4: Machine Learning Models
- [ ] Linear regression per zipcode
- [ ] Premium prediction model
- [ ] Feature importance analysis

## Setup Instructions

### Prerequisites
- Python 3.8+
- Git
- DVC (for Task 2)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Insurance-risk-analytics-end-to-end
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Initialize DVC (for Task 2):
```bash
pip install dvc
dvc init
```

## Key Metrics & KPIs

### Loss Ratio
```
Loss Ratio = TotalClaims / TotalPremium
```

### Analysis Focus Areas
- Loss ratio by Province, VehicleType, and Gender
- Distribution of financial variables
- Temporal trends (18-month period)
- Vehicle make/model risk analysis

## Learning Outcomes

- Data Engineering (DE)
- Predictive Analytics (PA)
- Machine Learning Engineering (MLE)
- Statistical Modeling
- A/B Testing Design
- Data Versioning
- Modular Python Development

## Team

- **Facilitators:** Kerod, Mahbubah, Filimon

## Key Dates

- **Challenge Introduction:** 10:30 AM UTC, Wednesday, 03 Dec 2025
- **Interim Submission:** 8:00 PM UTC, Sunday, 07 Dec 2025
- **Final Submission:** 8:00 PM UTC, Tuesday, 09 Dec 2025

## Deliverables

1. Comprehensive EDA report with visualizations
2. Statistical hypothesis testing results
3. Machine learning models with performance metrics
4. Feature importance analysis
5. Business recommendations
6. Reproducible codebase with version control

## License

This project is part of the KAIM Training Portfolio.

