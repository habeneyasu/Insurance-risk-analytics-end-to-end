# Insurance Risk Analytics & Predictive Modeling

End-to-end insurance risk analytics solution for **AlphaCare Insurance Solutions (ACIS)**, focused on car insurance planning and marketing in South Africa.

## Project Objective

The primary objective of this project is to analyze historical insurance claim data to:

1. **Optimize Marketing Strategies**: Identify high-value customer segments and develop targeted marketing approaches to maximize customer acquisition and retention
2. **Attract New Clients**: Discover low-risk segments where premiums can be strategically reduced to gain competitive advantage in the South African insurance market
3. **Portfolio Optimization**: Improve profitability by analyzing historical claim patterns and implementing data-driven risk-based pricing strategies

The project delivers actionable insights through comprehensive exploratory data analysis, statistical hypothesis testing, and machine learning models to support data-driven decision-making for ACIS.

## Project Architecture

### System Design

The project follows an **object-oriented, modular architecture** ensuring maintainability, scalability, and reproducibility:

```
Insurance-risk-analytics-end-to-end/
├── data/                      # Data files (versioned with DVC)
│   └── MachineLearningRating_v3.txt
├── notebooks/                 # Jupyter notebooks for interactive analysis
│   └── 01_eda_exploration.ipynb
├── src/                       # Source code (OOP-based)
│   ├── data/                  # Data layer
│   │   └── load_data.py      # DataLoader class
│   ├── analysis/              # Analysis layer
│   │   ├── data_quality.py   # DataQualityChecker class
│   │   ├── eda.py            # EDAAnalyzer class
│   │   ├── hypothesis_testing.py  # HypothesisTester class
│   │   └── visualizations.py # VisualizationGenerator class
│   └── models/                # Modeling layer
│       ├── data_preprocessing.py    # DataPreprocessor class
│       ├── model_builder.py         # ModelBuilder class
│       └── model_interpretability.py # ModelInterpreter class
├── reports/                    # Generated outputs
│   ├── figures/               # Visualizations
│   └── *.txt                  # Analysis reports
├── .github/workflows/          # CI/CD pipelines
└── requirements.txt           # Python dependencies
```

### Key Design Principles

- **Modularity**: Each component has a single, well-defined responsibility
- **Reproducibility**: DVC ensures data versioning; all analyses are scripted
- **Scalability**: OOP design allows easy extension and modification
- **Maintainability**: Clear code structure with comprehensive documentation

## Key Findings

### Portfolio Performance

- **Overall Loss Ratio: 1.0477** - The portfolio is currently unprofitable, with claims exceeding premiums by 4.77%
- **Average Margin: R-2.96** - Negative margin indicates need for premium optimization
- **Claim Frequency: 0.28%** - Very low claim frequency, but high severity when claims occur (R23,273.39)

### Risk Segmentation

#### High-Risk Segments
- **Gauteng Province**: Loss ratio of 1.22 (highest risk province)
- **Heavy Commercial Vehicles**: Loss ratio of 1.63 (very high risk)
- **KwaZulu-Natal & Western Cape**: Loss ratios above 1.0 (unprofitable)

#### Low-Risk Opportunities
- **Female Drivers**: Loss ratio of 0.82 (lowest risk demographic)
- **Light Commercial Vehicles**: Loss ratio of 0.23 (highly profitable)
- **Bus Category**: Loss ratio of 0.14 (very profitable)
- **Mpumalanga & North West Provinces**: Loss ratios 0.72-0.79 (profitable segments)

### Statistical Hypothesis Testing Results

The hypothesis testing implementation uses concrete test functions that construct control/test groups, run appropriate statistical tests (chi-squared, t-tests, Mann-Whitney U, Kruskal-Wallis), compute p-values, and return both statistical results and business conclusions.

**Test Implementation Details:**
- **Metrics Used**: Claim Frequency (proportion of policies with claims), Claim Severity (average claim amount given a claim occurred), and Margin (TotalPremium - TotalClaims)
- **Statistical Tests**: Chi-square tests for categorical/frequency data, Mann-Whitney U and Kruskal-Wallis tests for continuous/non-parametric comparisons
- **Data Segmentation**: Control and test groups are created with equivalence checks to ensure groups are comparable on other features
- **Significance Level**: α = 0.05

**Results:**

1. **Province Risk Differences**: **REJECTED H₀** (p < 0.000001)
   - Test: Chi-square (Frequency) + Kruskal-Wallis (Severity) across all provinces
   - Significant risk differences exist across provinces
   - Gauteng exhibits 332.3% higher loss ratio than Northern Cape (1.222 vs 0.283)
   - **Business Conclusion**: Regional risk adjustments to premiums are warranted

2. **Zipcode Risk Differences**: **REJECTED H₀** (p < 0.000001)
   - Test: Chi-square (Frequency) + Kruskal-Wallis (Severity) across 818 zip codes
   - Significant risk differences between zip codes
   - Tested across all zip codes with sufficient data (≥50 records each)
   - **Business Conclusion**: Zipcode-level risk adjustments to premiums may be warranted

3. **Zipcode Margin Differences**: **REJECTED H₀** (p < 0.000001)
   - Test: Mann-Whitney U test for margin comparison between zip codes
   - Significant profit margin differences exist between zip codes
   - Zipcode 1423 exhibits 111.3% higher margin than zipcode 1342 (R171.01 vs R-1,511.89)
   - **Business Conclusion**: Zipcode-level profitability adjustments may be warranted

4. **Gender Risk Differences**: **FAILED TO REJECT H₀** (p = 0.224)
   - Test: Chi-square (Frequency) + Mann-Whitney U (Severity)
   - No statistically significant risk differences between men and women
   - Frequency p-value: 0.951, Severity p-value: 0.224
   - **Business Conclusion**: Gender should not be a primary factor in pricing decisions, aligning with fair insurance practices

### Machine Learning Model Performance

- **Claim Severity Models**: Regression models (Linear, Random Forest, XGBoost) trained on policies with claims
- **Premium Optimization Models**: Predictive models for optimal premium calculation
- **Claim Probability Models**: Classification models to predict claim likelihood
- **Model Interpretability**: SHAP analysis identifies top 5-10 most influential features

## Business Recommendations

### 1. Premium Reduction Strategy

**Target Low-Risk Segments for Competitive Pricing**:
- **Female Drivers**: Offer 10-15% premium reduction to attract new female clients (loss ratio: 0.82)
- **Light Commercial Vehicles**: Develop specialized marketing campaigns with competitive rates (loss ratio: 0.23)
- **Bus Category**: Create targeted commercial vehicle insurance products (loss ratio: 0.14)
- **Low-Risk Provinces**: Regional marketing initiatives in Mpumalanga and North West

**Expected Impact**: Attract new low-risk clients while maintaining profitability

### 2. Risk Mitigation Actions

**Address High-Risk Segments**:
- **Heavy Commercial Vehicles**: Review and adjust pricing strategy (loss ratio: 1.63 requires immediate attention)
- **High-Risk Provinces**: Implement regional risk adjustments for Gauteng, KwaZulu-Natal, and Western Cape
- **Geographic Segmentation**: Use zipcode-level risk analysis for granular pricing

**Expected Impact**: Improve overall portfolio profitability by 3-5%

### 3. Data-Driven Pricing Model

**Implement Risk-Based Premium Calculation**:
- Use ML models to predict claim probability and severity
- Premium = (Predicted Probability × Predicted Severity) + Expense Loading + Profit Margin
- Leverage SHAP insights to adjust premiums based on key risk factors

**Expected Impact**: Optimize pricing to improve loss ratio from 1.05 to below 1.0

### 4. Marketing Strategy Optimization

**Segmentation-Based Campaigns**:
- Develop province-specific marketing strategies
- Create vehicle-type-specific insurance products
- Use predictive models to identify high-value prospects

**Expected Impact**: Increase customer acquisition by 15-20% in low-risk segments

## Results Discussion

### Exploratory Data Analysis (Task 1)

The EDA revealed critical insights about portfolio composition and risk distribution. The overall loss ratio of 1.0477 indicates the portfolio is unprofitable, requiring immediate intervention. Geographic analysis shows significant variation in risk across provinces, with Gauteng being the highest risk region. Vehicle type analysis identifies Heavy Commercial as extremely high-risk, while Light Commercial and Bus categories represent profitable opportunities.

**Key Insight**: The portfolio has clear segmentation opportunities that can be leveraged for both risk mitigation and customer acquisition.

### Hypothesis Testing (Task 3)

**Implementation Approach:**

The A/B hypothesis testing implementation follows a rigorous statistical framework with concrete test functions in the `HypothesisTester` class (`src/analysis/hypothesis_testing.py`). Each hypothesis test:

1. **Selects Metrics**: Uses Claim Frequency, Claim Severity, and Margin as key performance indicators
2. **Creates Data Segmentation**: Constructs control and test groups, ensuring statistical equivalence on other features (client attributes, vehicle properties, insurance plan types) before testing
3. **Runs Statistical Tests**: 
   - Chi-square tests for categorical/frequency comparisons
   - Mann-Whitney U tests for two-group non-parametric comparisons
   - Kruskal-Wallis tests for multiple-group non-parametric comparisons
4. **Computes P-values**: All tests return p-values for hypothesis decision-making (α = 0.05)
5. **Generates Business Conclusions**: Each test provides both statistical results and actionable business interpretations

**Key Findings:**

Statistical testing confirmed significant risk differences across provinces and zip codes, validating the need for geographic-based pricing strategies. The rejection of null hypotheses for province, zipcode risk, and zipcode margin differences provides strong evidence for implementing location-based premium adjustments.

**Key Insight**: Geographic factors are significant risk drivers, supporting the implementation of region-specific pricing models. The failure to reject the null hypothesis for gender-based differences is important - it suggests that gender should not be a primary factor in pricing decisions, aligning with fair insurance practices.

**Test Execution:**
- All tests are executed via `python src/run_hypothesis_tests.py`
- Comprehensive reports are generated in `reports/hypothesis_testing_report.txt`
- Each test includes detailed statistical outputs, p-values, and business recommendations

### Machine Learning Models (Task 4)

The ML models provide predictive capabilities for:
1. **Claim Severity**: Enables accurate estimation of financial liability for policies with claims
2. **Premium Optimization**: Supports data-driven premium calculation based on risk factors
3. **Claim Probability**: Helps identify high-risk policies before claims occur

SHAP analysis reveals the most influential features driving predictions, providing actionable insights for premium adjustments and risk assessment.

**Key Insight**: ML models enable dynamic, risk-based pricing that can improve portfolio profitability while remaining competitive in the market.

### Overall Impact

The comprehensive analysis provides ACIS with:
- **Data-Driven Insights**: Clear identification of profitable and unprofitable segments
- **Actionable Recommendations**: Specific strategies for premium optimization and marketing
- **Predictive Capabilities**: ML models for ongoing risk assessment and pricing
- **Competitive Advantage**: Ability to offer competitive rates to low-risk segments while maintaining profitability

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/habeneyasu/Insurance-risk-analytics-end-to-end.git
cd Insurance-risk-analytics-end-to-end

# Setup environment
chmod +x setup.sh
./setup.sh
```

### Running Analyses

```bash
# Exploratory Data Analysis
python src/run_eda.py

# Hypothesis Testing
python src/run_hypothesis_tests.py

# Machine Learning Models
python src/run_ml_models.py
```

### Pull Request Workflow

This project follows a **branch-based PR workflow** to ensure code quality and maintainability:

1. **Create Topic-Focused Branches**: For each task or feature, create a dedicated branch from `main`
   ```bash
   git checkout -b task-3-hypothesis-testing-update
   ```

2. **Make Changes and Commit**: Implement your changes with descriptive commit messages
   ```bash
   git add .
   git commit -m "Task 3: Implement comprehensive A/B hypothesis testing"
   ```

3. **Push Branch and Create PR**: Push your branch to remote and open a Pull Request
   ```bash
   git push origin task-3-hypothesis-testing-update
   ```
   Then create a PR on GitHub from your branch to `main`.

4. **CI Checks**: The PR will automatically trigger CI checks:
   - Linting with flake8
   - Code formatting check with black
   - Test execution
   - PR validation

5. **Review and Merge**: After CI passes and code review, merge the PR into `main`.

**Task Branches:**
- `task-1`: Git, GitHub, and EDA
- `task-2`: Data Version Control (DVC)
- `task-3`: A/B Hypothesis Testing
- `task-4`: Machine Learning Models

**Best Practices:**
- Use smaller, topic-focused branches for better traceability
- Link PRs to issues when applicable
- Ensure all CI checks pass before merging
- Write descriptive commit messages and PR descriptions

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

## Dataset

- **Size**: 1,000,098 records
- **Features**: 52 columns
- **Time Period**: October 2013 - August 2015 (23 months)
- **Unique Policies**: 7,000
- **Provinces**: 9 South African provinces

## Team

Facilitators: Kerod, Mahbubah, Filimon

## License

Part of the KAIM Training Portfolio.
