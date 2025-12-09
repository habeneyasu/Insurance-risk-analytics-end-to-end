# Task 3: Hypothesis Testing Code Evidence

This document provides clear evidence of the hypothesis testing implementation with concrete test functions.

## Code Files Location

### Main Implementation Files:
1. **`src/analysis/hypothesis_testing.py`** (845 lines)
   - Contains the `HypothesisTester` class with all concrete test functions
   - Defines KPIs, constructs control/test groups, runs statistical tests, computes p-values

2. **`src/run_hypothesis_tests.py`** (86 lines)
   - Main execution script that calls all test functions
   - Demonstrates usage of the concrete test functions

3. **`examples/hypothesis_testing_example.py`**
   - Example script showing how to use the test functions

## Concrete Test Functions

All functions are in `src/analysis/hypothesis_testing.py`:

### 1. `test_province_risk_differences(alpha=0.05)`
- **Location**: Lines 127-236
- **What it does**:
  - Defines KPI: Claim Frequency and Claim Severity
  - Constructs control/test groups across all provinces
  - Runs Chi-square test for frequency (line 161)
  - Runs Kruskal-Wallis test for severity (line 164)
  - Computes p-values (lines 168-169)
  - Returns statistical results + business conclusion (lines 215-226)

### 2. `test_zipcode_risk_differences(alpha=0.05)`
- **Location**: Lines 238-359
- **What it does**:
  - Defines KPI: Claim Frequency and Claim Severity
  - Constructs control/test groups across 818 zip codes (lines 258-260)
  - Runs Chi-square test for frequency (line 275)
  - Runs Kruskal-Wallis test for severity (line 279)
  - Computes p-values (lines 283-284)
  - Returns statistical results + business conclusion (lines 337-349)

### 3. `test_zipcode_margin_differences(alpha=0.05)`
- **Location**: Lines 368-490
- **What it does**:
  - Defines KPI: Margin (TotalPremium - TotalClaims)
  - Constructs control/test groups (highest vs lowest margin zip codes) (lines 381-390)
  - Checks group equivalence (lines 393-397)
  - Runs Mann-Whitney U test for margin comparison (line 408)
  - Computes p-value (line 409)
  - Returns statistical results + business conclusion (lines 440-452)

### 4. `test_gender_risk_differences(alpha=0.05)`
- **Location**: Lines 492-653
- **What it does**:
  - Defines KPI: Claim Frequency and Claim Severity
  - Constructs control/test groups (Female vs Male) (lines 530-531)
  - Checks group equivalence (lines 534-538)
  - Runs Chi-square test for frequency (line 543)
  - Runs Mann-Whitney U test for severity (line 560)
  - Computes p-values (lines 575-576)
  - Returns statistical results + business conclusion (lines 612-624)

## Helper Functions

### `_test_categorical_frequency(group_col, outcome_col, test_name)`
- **Location**: Lines 662-687
- **What it does**: Runs Chi-square test for frequency data
- **Returns**: p-value, chi-square statistic, degrees of freedom

### `_test_categorical_severity(group_col, severity_col, test_name)`
- **Location**: Lines 696-720
- **What it does**: Runs Kruskal-Wallis test for severity data
- **Returns**: p-value, test statistic, number of groups

### `_check_group_equivalence(group_a, group_b, feature_col)`
- **Location**: Lines 60-125
- **What it does**: Verifies control and test groups are equivalent on other features
- **Returns**: Equivalence test results for each feature

## KPI Definition

KPIs are defined in `_prepare_metrics()` method (lines 42-58):
- **Claim Frequency**: `self.data['HasClaim'] = (self.data['TotalClaims'] > 0).astype(int)`
- **Claim Severity**: `self.data['ClaimSeverityGivenClaim']` (average claim amount given claim occurred)
- **Margin**: `self.data['Margin'] = self.data['TotalPremium'] - self.data['TotalClaims']`

## Usage Example

```python
from src.data.load_data import DataLoader
from src.analysis.hypothesis_testing import HypothesisTester

# Load data
data_loader = DataLoader(data_dir="data")
df = data_loader.load_data(file_name="MachineLearningRating_v3.txt")

# Initialize tester (defines KPIs)
tester = HypothesisTester(df)

# Run concrete test functions
result1 = tester.test_province_risk_differences()      # Returns dict with p-values and conclusions
result2 = tester.test_zipcode_risk_differences()    # Returns dict with p-values and conclusions
result3 = tester.test_zipcode_margin_differences()   # Returns dict with p-values and conclusions
result4 = tester.test_gender_risk_differences()      # Returns dict with p-values and conclusions

# Or run all tests at once
all_results = tester.run_all_tests()  # Line 727
```

## Statistical Tests Used

1. **Chi-square test** (`chi2_contingency` from scipy.stats)
   - Used for: Claim frequency comparisons
   - Location: Lines 161, 275, 543

2. **Kruskal-Wallis test** (`stats.kruskal` from scipy.stats)
   - Used for: Multiple group severity comparisons
   - Location: Lines 164, 279

3. **Mann-Whitney U test** (`mannwhitneyu` from scipy.stats)
   - Used for: Two-group severity and margin comparisons
   - Location: Lines 408, 560

## Output Generation

All test functions:
- Print detailed test execution information
- Return dictionaries with:
  - `p_value` or `p_value_frequency` and `p_value_severity`
  - `reject_null` (boolean)
  - `business_conclusion` (string with interpretation)
  - Test statistics and group information

Reports are generated by `generate_report()` method (lines 732-808) and saved to `reports/hypothesis_testing_report.txt`.

## Verification

To verify the code works:
```bash
python src/run_hypothesis_tests.py
```

This will:
1. Load data
2. Initialize HypothesisTester (defines KPIs)
3. Run all 4 concrete test functions
4. Generate report with p-values and business conclusions
```

