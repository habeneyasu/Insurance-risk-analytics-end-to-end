"""
Main script for running A/B Hypothesis Testing (Task 3).
Uses object-oriented HypothesisTester class for modular and maintainable code.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from data.load_data import DataLoader
from analysis.hypothesis_testing import HypothesisTester


def main():
    """Main function to run hypothesis testing pipeline."""
    
    print("="*70)
    print("INSURANCE RISK ANALYTICS - A/B HYPOTHESIS TESTING")
    print("AlphaCare Insurance Solutions (ACIS) - Task 3")
    print("="*70)
    
    # Step 1: Load Data
    print("\n[Step 1] Loading data...")
    data_loader = DataLoader(data_dir="data")
    df = data_loader.load_data(file_name="MachineLearningRating_v3.txt")
    print(f"  ✓ Data loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
    
    # Step 2: Initialize Hypothesis Tester
    print("\n[Step 2] Initializing Hypothesis Tester...")
    tester = HypothesisTester(df)
    print("  ✓ Hypothesis Tester initialized")
    print("  ✓ Metrics prepared:")
    print("    - Claim Frequency (proportion of policies with at least one claim)")
    print("    - Claim Severity (average claim amount given a claim occurred)")
    print("    - Margin (TotalPremium - TotalClaims)")
    
    # Step 3: Run All Hypothesis Tests
    print("\n[Step 3] Running all hypothesis tests...")
    print("\n" + "="*70)
    print("HYPOTHESIS TESTS TO EXECUTE:")
    print("="*70)
    print("1. H₀: There are no risk differences across provinces")
    print("2. H₀: There are no risk differences between zip codes")
    print("3. H₀: There is no significant margin (profit) difference between zip codes")
    print("4. H₀: There is no significant risk difference between Women and Men")
    print("="*70)
    
    # Run all tests with alpha = 0.05
    alpha = 0.05
    all_results = tester.run_all_tests(alpha=alpha)
    
    # Step 4: Generate Report
    print("\n[Step 4] Generating comprehensive report...")
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    
    report_path = reports_dir / "hypothesis_testing_report.txt"
    report_content = tester.generate_report(output_path=str(report_path))
    
    print("\n" + "="*70)
    print("HYPOTHESIS TESTING COMPLETE")
    print("="*70)
    print(f"\nReport saved to: {report_path}")
    print("\nSummary of Results:")
    print("-"*70)
    
    for test_name, result in all_results.items():
        if 'status' not in result:
            status = "REJECT H₀" if result.get('reject_null', False) else "FAIL TO REJECT H₀"
            p_val = result.get('p_value', result.get('overall_p_value', 'N/A'))
            if isinstance(p_val, (int, float)):
                print(f"  {test_name.replace('_', ' ').title()}: {status} (p = {p_val:.6f})")
            else:
                print(f"  {test_name.replace('_', ' ').title()}: {status}")
    
    print("\n" + "="*70)
    print("For detailed results and business conclusions, see the report file.")
    print("="*70)


if __name__ == "__main__":
    main()

