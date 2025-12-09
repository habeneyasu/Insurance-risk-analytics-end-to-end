"""
Example: A/B Hypothesis Testing Implementation
==============================================

This example demonstrates the concrete test functions that:
1. Define KPIs (Claim Frequency, Claim Severity, Margin)
2. Construct control/test groups
3. Run chi-square/t-tests
4. Compute p-values
5. Generate reported outputs with clear, callable functions

All functions are in src/analysis/hypothesis_testing.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.load_data import DataLoader
from src.analysis.hypothesis_testing import HypothesisTester


def example_hypothesis_testing():
    """
    Example demonstrating concrete hypothesis testing functions.
    
    This shows:
    - KPI definition (Claim Frequency, Claim Severity, Margin)
    - Control/test group construction
    - Statistical tests (chi-square, Mann-Whitney U, Kruskal-Wallis)
    - P-value computation
    - Business conclusions
    """
    
    print("="*70)
    print("HYPOTHESIS TESTING - CONCRETE TEST FUNCTIONS EXAMPLE")
    print("="*70)
    
    # Load data
    print("\n[1] Loading data...")
    data_loader = DataLoader(data_dir="data")
    df = data_loader.load_data(file_name="MachineLearningRating_v3.txt")
    print(f"   ✓ Loaded {df.shape[0]:,} records")
    
    # Initialize HypothesisTester (defines KPIs and prepares metrics)
    print("\n[2] Initializing HypothesisTester...")
    print("   This class defines KPIs:")
    print("   - Claim Frequency: proportion of policies with at least one claim")
    print("   - Claim Severity: average claim amount given a claim occurred")
    print("   - Margin: TotalPremium - TotalClaims")
    tester = HypothesisTester(df)
    
    # Demonstrate concrete test functions
    print("\n[3] CONCRETE TEST FUNCTIONS AVAILABLE:")
    print("   " + "-"*66)
    print("   Function: test_province_risk_differences()")
    print("   - Constructs control/test groups across all provinces")
    print("   - Runs Chi-square test for frequency")
    print("   - Runs Kruskal-Wallis test for severity")
    print("   - Computes p-values")
    print("   - Returns statistical results + business conclusion")
    print()
    print("   Function: test_zipcode_risk_differences()")
    print("   - Constructs control/test groups across zip codes")
    print("   - Runs Chi-square test for frequency")
    print("   - Runs Kruskal-Wallis test for severity")
    print("   - Computes p-values")
    print("   - Returns statistical results + business conclusion")
    print()
    print("   Function: test_zipcode_margin_differences()")
    print("   - Constructs control/test groups (highest vs lowest margin zip codes)")
    print("   - Runs Mann-Whitney U test for margin comparison")
    print("   - Computes p-value")
    print("   - Returns statistical results + business conclusion")
    print()
    print("   Function: test_gender_risk_differences()")
    print("   - Constructs control/test groups (Female vs Male)")
    print("   - Runs Chi-square test for frequency")
    print("   - Runs Mann-Whitney U test for severity")
    print("   - Computes p-values")
    print("   - Returns statistical results + business conclusion")
    
    # Run one test as example
    print("\n[4] EXAMPLE: Running test_province_risk_differences()")
    print("   " + "-"*66)
    result = tester.test_province_risk_differences(alpha=0.05)
    
    print("\n[5] RESULT STRUCTURE:")
    print("   " + "-"*66)
    print(f"   - Hypothesis: {result['hypothesis']}")
    print(f"   - Test Type: {result['test_type']}")
    print(f"   - P-value (Frequency): {result['p_value_frequency']:.6f}")
    print(f"   - P-value (Severity): {result['p_value_severity']:.6f}")
    print(f"   - Overall P-value: {result['overall_p_value']:.6f}")
    print(f"   - Reject Null: {result['reject_null']}")
    print(f"   - Business Conclusion: {result['business_conclusion'][:100]}...")
    
    print("\n" + "="*70)
    print("All test functions are callable and return:")
    print("  ✓ Statistical test results (p-values, test statistics)")
    print("  ✓ Business conclusions and interpretations")
    print("  ✓ Control/test group information")
    print("="*70)


if __name__ == "__main__":
    example_hypothesis_testing()

