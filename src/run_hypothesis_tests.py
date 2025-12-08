"""
Main script for running Hypothesis Testing (Task 3).
Tests all null hypotheses and generates comprehensive report with business recommendations.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from data.load_data import DataLoader
from analysis.hypothesis_testing import HypothesisTester
import pandas as pd


def format_business_recommendation(test_name: str, result: dict, metric: str) -> str:
    """
    Format business recommendation based on test results.
    
    Parameters:
    -----------
    test_name : str
        Name of the test
    result : dict
        Test result dictionary
    metric : str
        Metric being tested
    
    Returns:
    --------
    str
        Formatted business recommendation
    """
    if 'error' in result:
        return f"Error in {metric} test: {result['error']}"
    
    if not result.get('reject_null', False):
        return f"✓ No significant difference in {metric} ({test_name}). No action required."
    
    p_value = result.get('p_value', 1.0)
    interpretation = "highly significant" if p_value < 0.01 else "significant"
    
    if 'mean_difference' in result:
        group_a = result.get('group_a', 'Group A')
        group_b = result.get('group_b', 'Group B')
        mean_a = result.get('group_a_mean', 0)
        mean_b = result.get('group_b_mean', 0)
        percent_diff = result.get('percent_difference', 0)
        
        direction = "higher" if percent_diff > 0 else "lower"
        
        return (
            f"⚠️ REJECT H₀: {group_a} shows {abs(percent_diff):.1f}% {direction} {metric} "
            f"than {group_b} (p < {p_value:.4f}, {interpretation}). "
            f"Consider adjusting premiums or risk assessment for {group_a}."
        )
    elif 'group_means' in result:
        group_means = result.get('group_means', {})
        if group_means:
            highest = max(group_means.items(), key=lambda x: x[1])
            lowest = min(group_means.items(), key=lambda x: x[1])
            diff_pct = ((highest[1] - lowest[1]) / lowest[1] * 100) if lowest[1] != 0 else 0
            
            return (
                f"⚠️ REJECT H₀: Significant differences in {metric} across groups "
                f"(p < {p_value:.4f}, {interpretation}). {highest[0]} shows {diff_pct:.1f}% "
                f"higher {metric} than {lowest[0]}. Consider segment-specific pricing."
            )
    
    return f"⚠️ REJECT H₀: Significant difference in {metric} detected (p < {p_value:.4f})."


def main():
    """Main function to run all hypothesis tests."""
    
    print("=" * 80)
    print("HYPOTHESIS TESTING - INSURANCE RISK ANALYTICS")
    print("AlphaCare Insurance Solutions (ACIS)")
    print("=" * 80)
    
    # Step 1: Load Data
    print("\n[Step 1] Loading data...")
    data_loader = DataLoader(data_dir="data")
    df = data_loader.load_data(file_name="MachineLearningRating_v3.txt")
    print(f"Data loaded: {df.shape[0]:,} records, {df.shape[1]} features")
    
    # Step 2: Initialize Hypothesis Tester
    print("\n[Step 2] Initializing hypothesis testing framework...")
    tester = HypothesisTester(df)
    
    # Step 3: Test H₀: No risk differences across provinces
    print("\n[Step 3] Testing H₀: No risk differences across provinces...")
    province_results = tester.test_province_risk_differences()
    print("  ✓ Province risk tests completed")
    
    # Step 4: Test H₀: No risk differences between zip codes
    print("\n[Step 4] Testing H₀: No risk differences between zip codes...")
    zipcode_risk_results = tester.test_zipcode_risk_differences(top_n=10)
    print("  ✓ Zipcode risk tests completed")
    
    # Step 5: Test H₀: No margin difference between zip codes
    print("\n[Step 5] Testing H₀: No margin (profit) difference between zip codes...")
    zipcode_margin_results = tester.test_zipcode_margin_differences(top_n=10)
    print("  ✓ Zipcode margin tests completed")
    
    # Step 6: Test H₀: No risk difference between Women and Men
    print("\n[Step 6] Testing H₀: No risk difference between Women and Men...")
    gender_results = tester.test_gender_risk_differences()
    print("  ✓ Gender risk tests completed")
    
    # Step 7: Generate Report
    print("\n[Step 7] Generating comprehensive report...")
    report = tester.generate_report()
    
    # Step 8: Business Recommendations
    print("\n" + "=" * 80)
    print("BUSINESS RECOMMENDATIONS")
    print("=" * 80)
    
    recommendations = []
    
    # Province recommendations
    if 'province_risk' in tester.results:
        prov_results = tester.results['province_risk']
        if 'claim_frequency' in prov_results:
            rec = format_business_recommendation("Provinces", prov_results['claim_frequency'], "Claim Frequency")
            recommendations.append(("Province Risk - Claim Frequency", rec))
        
        if 'claim_severity' in prov_results:
            rec = format_business_recommendation("Provinces", prov_results['claim_severity'], "Claim Severity")
            recommendations.append(("Province Risk - Claim Severity", rec))
    
    # Zipcode risk recommendations
    if 'zipcode_risk' in tester.results:
        zip_results = tester.results['zipcode_risk']
        if 'claim_frequency' in zip_results:
            rec = format_business_recommendation("Zipcodes", zip_results['claim_frequency'], "Claim Frequency")
            recommendations.append(("Zipcode Risk - Claim Frequency", rec))
        
        if 'claim_severity' in zip_results:
            rec = format_business_recommendation("Zipcodes", zip_results['claim_severity'], "Claim Severity")
            recommendations.append(("Zipcode Risk - Claim Severity", rec))
    
    # Zipcode margin recommendations
    if 'zipcode_margin' in tester.results:
        margin_results = tester.results['zipcode_margin']
        if 'margin' in margin_results:
            rec = format_business_recommendation("Zipcodes", margin_results['margin'], "Margin")
            recommendations.append(("Zipcode Margin", rec))
    
    # Gender recommendations
    if 'gender_risk' in tester.results:
        gen_results = tester.results['gender_risk']
        if 'claim_frequency' in gen_results:
            rec = format_business_recommendation("Gender", gen_results['claim_frequency'], "Claim Frequency")
            recommendations.append(("Gender Risk - Claim Frequency", rec))
        
        if 'claim_severity' in gen_results:
            rec = format_business_recommendation("Gender", gen_results['claim_severity'], "Claim Severity")
            recommendations.append(("Gender Risk - Claim Severity", rec))
    
    # Print recommendations
    for title, recommendation in recommendations:
        print(f"\n{title}:")
        print(f"  {recommendation}")
    
    # Step 9: Save detailed report
    print("\n" + "=" * 80)
    print("DETAILED STATISTICAL REPORT")
    print("=" * 80)
    print(report)
    
    # Save report to file
    report_file = Path("reports") / "hypothesis_testing_report.txt"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_file, 'w') as f:
        f.write("HYPOTHESIS TESTING REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(report)
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("BUSINESS RECOMMENDATIONS\n")
        f.write("=" * 80 + "\n\n")
        for title, recommendation in recommendations:
            f.write(f"{title}:\n")
            f.write(f"  {recommendation}\n\n")
    
    print(f"\n✓ Detailed report saved to: {report_file}")
    
    # Step 10: Summary Statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    # Overall metrics
    overall_freq = tester.calculate_claim_frequency()
    overall_severity = tester.calculate_claim_severity()
    overall_margin = tester.calculate_margin()
    
    print(f"\nOverall Portfolio Metrics:")
    print(f"  Claim Frequency: {overall_freq:.4f} ({overall_freq*100:.2f}%)")
    print(f"  Claim Severity: R{overall_severity:.2f}")
    print(f"  Average Margin: R{overall_margin:.2f}")
    
    print("\n" + "=" * 80)
    print("Hypothesis testing completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()

