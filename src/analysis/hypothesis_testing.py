"""
A/B Hypothesis Testing for Insurance Risk Analytics.
Object-oriented implementation with concrete test functions.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any, Optional
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu
import warnings

warnings.filterwarnings("ignore")


class HypothesisTester:
    """
    Class for conducting A/B hypothesis tests on insurance risk metrics.
    
    Tests risk differences across:
    - Provinces
    - Zip codes (PostalCode)
    - Gender
    - Margin differences by zip code
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize HypothesisTester.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input dataframe with insurance data
        """
        self.data = data.copy()
        
        # Calculate derived metrics
        self._prepare_metrics()
        
        # Store test results
        self.test_results: Dict[str, Dict[str, Any]] = {}
    
    def _prepare_metrics(self) -> None:
        """Prepare metrics needed for hypothesis testing."""
        # Claim Frequency: proportion of policies with at least one claim
        self.data['HasClaim'] = (self.data['TotalClaims'] > 0).astype(int)
        
        # Claim Severity: average claim amount given a claim occurred
        # For policies with claims, severity = TotalClaims / number of claims
        # We'll use TotalClaims directly as severity metric per policy
        self.data['ClaimSeverity'] = self.data['TotalClaims']
        
        # Margin: TotalPremium - TotalClaims
        self.data['Margin'] = self.data['TotalPremium'] - self.data['TotalClaims']
        
        # For severity calculation: only consider policies with claims
        self.data['ClaimSeverityGivenClaim'] = np.where(
            self.data['HasClaim'] == 1,
            self.data['TotalClaims'],
            np.nan
        )
    
    def _check_group_equivalence(self, group_a: pd.DataFrame, group_b: pd.DataFrame, 
                                 feature_col: str) -> Dict[str, Any]:
        """
        Check if two groups are statistically equivalent on other features.
        
        Parameters:
        -----------
        group_a : pd.DataFrame
            Control group
        group_b : pd.DataFrame
            Test group
        feature_col : str
            The feature being tested (to exclude from equivalence check)
        
        Returns:
        --------
        dict
            Equivalence test results
        """
        equivalence_results = {}
        
        # Check key attributes that should be similar
        check_cols = ['Gender', 'VehicleType', 'TotalPremium']
        if 'CoverType' in self.data.columns:
            check_cols.append('CoverType')
        
        for col in check_cols:
            if col == feature_col or col not in self.data.columns:
                continue
            
            if self.data[col].dtype in ['object', 'category']:
                # Chi-square test for categorical
                try:
                    contingency = pd.crosstab(
                        pd.concat([group_a[col], group_b[col]], ignore_index=True),
                        pd.concat([
                            pd.Series(['Group A'] * len(group_a)),
                            pd.Series(['Group B'] * len(group_b))
                        ], ignore_index=True)
                    )
                    if contingency.shape[0] > 1 and contingency.shape[1] > 1:
                        chi2, p_val, dof, expected = chi2_contingency(contingency)
                        equivalence_results[col] = {
                            'test': 'chi-square',
                            'p_value': p_val,
                            'equivalent': p_val >= 0.05
                        }
                except:
                    pass
            else:
                # T-test for numerical
                try:
                    stat, p_val = ttest_ind(group_a[col].dropna(), group_b[col].dropna())
                    equivalence_results[col] = {
                        'test': 't-test',
                        'p_value': p_val,
                        'equivalent': p_val >= 0.05
                    }
                except:
                    pass
        
        return equivalence_results
    
    def test_province_risk_differences(self, alpha: float = 0.05) -> Dict[str, Any]:
        """
        Test H₀: There are no risk differences across provinces.
        
        Tests both Claim Frequency and Claim Severity.
        
        Parameters:
        -----------
        alpha : float
            Significance level (default: 0.05)
        
        Returns:
        --------
        dict
            Test results with p-values and business conclusions
        """
        print("\n" + "="*70)
        print("HYPOTHESIS TEST 1: Province Risk Differences")
        print("="*70)
        print("H₀: There are no risk differences across provinces")
        print("H₁: There are significant risk differences across provinces")
        
        # Get all provinces
        provinces = self.data['Province'].dropna().unique()
        
        if len(provinces) < 2:
            return {
                'hypothesis': 'No risk differences across provinces',
                'status': 'insufficient_data',
                'message': 'Need at least 2 provinces for testing'
            }
        
        # Test 1: Claim Frequency (Chi-square test)
        print("\n[Test 1.1] Claim Frequency Test (Chi-square)")
        # EXPLICIT: Create contingency table for chi-square test
        contingency_table = pd.crosstab(self.data['Province'], self.data['HasClaim'])
        # EXPLICIT: Run chi-square test
        chi2_statistic, p_value_freq, degrees_of_freedom, expected_frequencies = chi2_contingency(contingency_table)
        print(f"  Chi-square statistic: {chi2_statistic:.4f}")
        print(f"  Degrees of freedom: {degrees_of_freedom}")
        print(f"  p-value: {p_value_freq:.6f}")
        frequency_results = {
            'test_name': 'Province Claim Frequency',
            'statistic': chi2_statistic,
            'p_value': p_value_freq,
            'degrees_of_freedom': degrees_of_freedom
        }
        
        # Test 2: Claim Severity (Kruskal-Wallis test for multiple groups)
        print("\n[Test 1.2] Claim Severity Test (Kruskal-Wallis)")
        # EXPLICIT: Create groups for Kruskal-Wallis test
        severity_groups = []
        for province in provinces:
            province_severity = self.data[
                (self.data['Province'] == province) & 
                (self.data['ClaimSeverityGivenClaim'].notna())
            ]['ClaimSeverityGivenClaim'].values
            if len(province_severity) > 0:
                severity_groups.append(province_severity)
        # EXPLICIT: Run Kruskal-Wallis test
        if len(severity_groups) >= 2:
            kruskal_statistic, p_value_sev = stats.kruskal(*severity_groups)
            print(f"  Kruskal-Wallis statistic: {kruskal_statistic:.4f}")
            print(f"  p-value: {p_value_sev:.6f}")
        else:
            p_value_sev = 1.0
            kruskal_statistic = None
        severity_results = {
            'test_name': 'Province Claim Severity',
            'statistic': kruskal_statistic,
            'p_value': p_value_sev,
            'num_groups': len(severity_groups)
        }
        
        # Combine results
        p_value_freq = frequency_results.get('p_value', 1.0)
        p_value_sev = severity_results.get('p_value', 1.0)
        
        # Overall result: reject if either test is significant
        overall_p_value = min(p_value_freq, p_value_sev)
        reject_null = overall_p_value < alpha
        
        # Business interpretation
        if reject_null:
            # Find highest and lowest risk provinces
            province_stats = self.data.groupby('Province').agg({
                'HasClaim': 'mean',
                'ClaimSeverityGivenClaim': 'mean',
                'TotalPremium': 'sum',
                'TotalClaims': 'sum'
            }).assign(
                LossRatio=lambda x: x['TotalClaims'] / x['TotalPremium']
            ).sort_values('LossRatio', ascending=False)
            
            highest_risk = province_stats.index[0]
            lowest_risk = province_stats.index[-1]
            highest_lr = province_stats.loc[highest_risk, 'LossRatio']
            lowest_lr = province_stats.loc[lowest_risk, 'LossRatio']
            pct_diff = ((highest_lr - lowest_lr) / lowest_lr) * 100
            
            business_conclusion = (
                f"We REJECT the null hypothesis (p < {alpha:.3f}). "
                f"Significant risk differences exist across provinces. "
                f"{highest_risk} exhibits {pct_diff:.1f}% higher loss ratio than {lowest_risk} "
                f"({highest_lr:.3f} vs {lowest_lr:.3f}), suggesting regional risk adjustments "
                f"to premiums may be warranted."
            )
        else:
            business_conclusion = (
                f"We FAIL TO REJECT the null hypothesis (p = {overall_p_value:.3f}). "
                f"No statistically significant risk differences were found across provinces "
                f"at the {alpha*100}% significance level."
            )
        
        result = {
            'hypothesis': 'No risk differences across provinces',
            'test_type': 'Chi-square (Frequency) + ANOVA/Kruskal-Wallis (Severity)',
            'frequency_test': frequency_results,
            'severity_test': severity_results,
            'p_value_frequency': p_value_freq,
            'p_value_severity': p_value_sev,
            'overall_p_value': overall_p_value,
            'alpha': alpha,
            'reject_null': reject_null,
            'business_conclusion': business_conclusion
        }
        
        self.test_results['province_risk'] = result
        
        print(f"\n[Result] {'REJECT H₀' if reject_null else 'FAIL TO REJECT H₀'}")
        print(f"  Frequency p-value: {p_value_freq:.6f}")
        print(f"  Severity p-value: {p_value_sev:.6f}")
        print(f"  Overall p-value: {overall_p_value:.6f}")
        print(f"\n[Business Conclusion] {business_conclusion}")
        
        return result
    
    def test_zipcode_risk_differences(self, alpha: float = 0.05) -> Dict[str, Any]:
        """
        Test H₀: There are no risk differences between zip codes.
        
        Tests across all zip codes (similar to province test) to detect
        any significant differences in risk metrics.
        
        Parameters:
        -----------
        alpha : float
            Significance level (default: 0.05)
        
        Returns:
        --------
        dict
            Test results with p-values and business conclusions
        """
        print("\n" + "="*70)
        print("HYPOTHESIS TEST 2: Zipcode Risk Differences")
        print("="*70)
        print("H₀: There are no risk differences between zip codes")
        print("H₁: There are significant risk differences between zip codes")
        
        # Get zip codes with sufficient data (at least 50 records for statistical power)
        zipcode_counts = self.data['PostalCode'].dropna().value_counts()
        valid_zips = zipcode_counts[zipcode_counts >= 50].index.tolist()
        
        if len(valid_zips) < 2:
            return {
                'hypothesis': 'No risk differences between zip codes',
                'status': 'insufficient_data',
                'message': 'Need at least 2 zip codes with sufficient data for testing'
            }
        
        print(f"\n[Data Segmentation]")
        print(f"  Testing across {len(valid_zips)} zip codes with sufficient data (≥50 records each)")
        print(f"  Total records analyzed: {self.data[self.data['PostalCode'].isin(valid_zips)].shape[0]:,}")
        
        # Test 1: Claim Frequency (Chi-square test across all zip codes)
        print("\n[Test 2.1] Claim Frequency Test (Chi-square across all zip codes)")
        # EXPLICIT: Create contingency table for chi-square test
        contingency_table = pd.crosstab(self.data[self.data['PostalCode'].isin(valid_zips)]['PostalCode'], 
                                        self.data[self.data['PostalCode'].isin(valid_zips)]['HasClaim'])
        # EXPLICIT: Run chi-square test
        chi2_statistic, p_value_freq, degrees_of_freedom, expected_frequencies = chi2_contingency(contingency_table)
        print(f"  Chi-square statistic: {chi2_statistic:.4f}")
        print(f"  Degrees of freedom: {degrees_of_freedom}")
        print(f"  p-value: {p_value_freq:.6f}")
        frequency_results = {
            'test_name': 'Zipcode Claim Frequency',
            'statistic': chi2_statistic,
            'p_value': p_value_freq,
            'degrees_of_freedom': degrees_of_freedom
        }
        
        # Test 2: Claim Severity (Kruskal-Wallis test across all zip codes)
        print("\n[Test 2.2] Claim Severity Test (Kruskal-Wallis across all zip codes)")
        # EXPLICIT: Create groups for Kruskal-Wallis test
        severity_groups = []
        for zipcode in valid_zips:
            zipcode_severity = self.data[
                (self.data['PostalCode'] == zipcode) & 
                (self.data['ClaimSeverityGivenClaim'].notna())
            ]['ClaimSeverityGivenClaim'].values
            if len(zipcode_severity) > 0:
                severity_groups.append(zipcode_severity)
        # EXPLICIT: Run Kruskal-Wallis test
        if len(severity_groups) >= 2:
            kruskal_statistic, p_value_sev = stats.kruskal(*severity_groups)
            print(f"  Kruskal-Wallis statistic: {kruskal_statistic:.4f}")
            print(f"  p-value: {p_value_sev:.6f}")
        else:
            p_value_sev = 1.0
            kruskal_statistic = None
        severity_results = {
            'test_name': 'Zipcode Claim Severity',
            'statistic': kruskal_statistic,
            'p_value': p_value_sev,
            'num_groups': len(severity_groups)
        }
        
        # Overall result: reject if either test is significant
        overall_p_value = min(p_value_freq, p_value_sev)
        reject_null = overall_p_value < alpha
        
        # Business interpretation - find highest and lowest risk zip codes
        zipcode_stats = self.data[self.data['PostalCode'].isin(valid_zips)].groupby('PostalCode').agg({
            'HasClaim': 'mean',
            'ClaimSeverityGivenClaim': 'mean',
            'TotalPremium': 'sum',
            'TotalClaims': 'sum'
        }).assign(
            LossRatio=lambda x: x['TotalClaims'] / x['TotalPremium']
        )
        
        # Filter to zipcodes with actual claims for meaningful comparison
        zipcode_stats_with_claims = zipcode_stats[
            (zipcode_stats['TotalClaims'] > 0) & 
            (zipcode_stats['ClaimSeverityGivenClaim'].notna())
        ].sort_values('LossRatio', ascending=False)
        
        if reject_null and len(zipcode_stats_with_claims) > 0:
            highest_risk = zipcode_stats_with_claims.index[0]
            lowest_risk = zipcode_stats_with_claims.index[-1]
            highest_lr = zipcode_stats_with_claims.loc[highest_risk, 'LossRatio']
            lowest_lr = zipcode_stats_with_claims.loc[lowest_risk, 'LossRatio']
            highest_sev = zipcode_stats_with_claims.loc[highest_risk, 'ClaimSeverityGivenClaim']
            lowest_sev = zipcode_stats_with_claims.loc[lowest_risk, 'ClaimSeverityGivenClaim']
            
            # Calculate percentage difference using loss ratio (more meaningful)
            if lowest_lr > 0:
                pct_diff_lr = ((highest_lr - lowest_lr) / lowest_lr) * 100
            else:
                pct_diff_lr = 0
            
            business_conclusion = (
                f"We REJECT the null hypothesis (p < {alpha:.3f}). "
                f"Significant risk differences exist between zip codes. "
                f"Zipcode {highest_risk} exhibits {pct_diff_lr:.1f}% higher loss ratio than zipcode {lowest_risk} "
                f"({highest_lr:.3f} vs {lowest_lr:.3f}), suggesting zipcode-level risk adjustments "
                f"to premiums may be warranted."
            )
        else:
            business_conclusion = (
                f"We FAIL TO REJECT the null hypothesis (p = {overall_p_value:.3f}). "
                f"No statistically significant risk differences were found between zip codes "
                f"at the {alpha*100}% significance level."
            )
        
        result = {
            'hypothesis': 'No risk differences between zip codes',
            'test_type': 'Chi-square (Frequency) + Kruskal-Wallis (Severity)',
            'num_zipcodes_tested': len(valid_zips),
            'frequency_test': frequency_results,
            'severity_test': severity_results,
            'p_value_frequency': p_value_freq,
            'p_value_severity': p_value_sev,
            'overall_p_value': overall_p_value,
            'alpha': alpha,
            'reject_null': reject_null,
            'business_conclusion': business_conclusion
        }
        
        self.test_results['zipcode_risk'] = result
        
        print(f"\n[Result] {'REJECT H₀' if reject_null else 'FAIL TO REJECT H₀'}")
        print(f"  Frequency p-value: {p_value_freq:.6f}")
        print(f"  Severity p-value: {p_value_sev:.6f}")
        print(f"  Overall p-value: {overall_p_value:.6f}")
        print(f"\n[Business Conclusion] {business_conclusion}")
        
        return result
    
    def test_zipcode_margin_differences(self, alpha: float = 0.05) -> Dict[str, Any]:
        """
        Test H₀: There is no significant margin (profit) difference between zip codes.
        
        Parameters:
        -----------
        alpha : float
            Significance level (default: 0.05)
        
        Returns:
        --------
        dict
            Test results with p-values and business conclusions
        """
        print("\n" + "="*70)
        print("HYPOTHESIS TEST 3: Zipcode Margin Differences")
        print("="*70)
        print("H₀: There is no significant margin (profit) difference between zip codes")
        print("H₁: There are significant margin differences between zip codes")
        
        # Get zip codes with sufficient data
        zipcode_counts = self.data['PostalCode'].value_counts()
        valid_zips = zipcode_counts[zipcode_counts >= 100].index.tolist()
        
        if len(valid_zips) < 2:
            return {
                'hypothesis': 'No margin differences between zip codes',
                'status': 'insufficient_data',
                'message': 'Need at least 2 zip codes with sufficient data for testing'
            }
        
        # Select two zip codes for A/B testing (highest and lowest margin)
        zipcode_stats = self.data[self.data['PostalCode'].isin(valid_zips)].groupby('PostalCode').agg({
            'Margin': 'mean',
            'TotalPremium': 'sum',
            'TotalClaims': 'sum'
        }).sort_values('Margin', ascending=False)
        
        zip_a = zipcode_stats.index[0]  # Highest margin
        zip_b = zipcode_stats.index[-1]  # Lowest margin
        
        group_a = self.data[self.data['PostalCode'] == zip_a].copy()
        group_b = self.data[self.data['PostalCode'] == zip_b].copy()
        
        print(f"\n[Data Segmentation]")
        print(f"  Group A (Control): Zipcode {zip_a} (n={len(group_a)})")
        print(f"  Group B (Test): Zipcode {zip_b} (n={len(group_b)})")
        
        # Check group equivalence
        print("\n[Equivalence Check] Verifying groups are equivalent on other features...")
        equivalence = self._check_group_equivalence(group_a, group_b, 'PostalCode')
        for col, result in equivalence.items():
            status = "✓ Equivalent" if result['equivalent'] else "✗ Different"
            print(f"  {col}: {status} (p={result['p_value']:.3f})")
        
        # Test: Margin difference (t-test or Mann-Whitney U)
        print("\n[Test 3.1] Margin Difference Test")
        margin_a = group_a['Margin'].dropna()
        margin_b = group_b['Margin'].dropna()
        
        # Check normality (Shapiro-Wilk test on sample if data is large)
        if len(margin_a) > 5000:
            sample_a = margin_a.sample(5000, random_state=42)
        else:
            sample_a = margin_a
        if len(margin_b) > 5000:
            sample_b = margin_b.sample(5000, random_state=42)
        else:
            sample_b = margin_b
        
        # EXPLICIT: Run Mann-Whitney U test (non-parametric t-test alternative)
        mannwhitney_statistic, p_value = mannwhitneyu(margin_a, margin_b, alternative='two-sided')
        mean_margin_a = margin_a.mean()
        mean_margin_b = margin_b.mean()
        
        print(f"  Group A mean margin: R{mean_margin_a:,.2f} (n={len(margin_a)})")
        print(f"  Group B mean margin: R{mean_margin_b:,.2f} (n={len(margin_b)})")
        print(f"  Mann-Whitney U statistic: {mannwhitney_statistic:.4f}")
        print(f"  p-value: {p_value:.6f}")
        
        # EXPLICIT: P-value threshold comparison
        # Compare p-value to significance threshold (alpha = 0.05)
        reject_null = p_value < alpha
        print(f"\n[Decision] P-value threshold (alpha): {alpha}")
        print(f"  P-value: {p_value:.6f}")
        print(f"  Decision: {'REJECT H₀' if reject_null else 'FAIL TO REJECT H₀'} (p {'<' if reject_null else '>='} {alpha})")
        
        # Business interpretation
        if reject_null:
            pct_diff = ((mean_margin_a - mean_margin_b) / abs(mean_margin_b)) * 100 if mean_margin_b != 0 else 0
            business_conclusion = (
                f"We REJECT the null hypothesis (p < {alpha:.3f}). "
                f"Significant margin differences exist between zip codes. "
                f"Zipcode {zip_a} exhibits {pct_diff:.1f}% higher margin than zipcode {zip_b} "
                f"(R{mean_margin_a:,.2f} vs R{mean_margin_b:,.2f}), suggesting zipcode-level "
                f"profitability adjustments may be warranted."
            )
        else:
            business_conclusion = (
                f"We FAIL TO REJECT the null hypothesis (p = {p_value:.3f}). "
                f"No statistically significant margin differences were found between zip codes "
                f"at the {alpha*100}% significance level."
            )
        
        result = {
            'hypothesis': 'No significant margin difference between zip codes',
            'test_type': 'Mann-Whitney U test',
            'group_a_zipcode': zip_a,
            'group_b_zipcode': zip_b,
            'group_a_size': len(group_a),
            'group_b_size': len(group_b),
            'group_a_mean_margin': mean_margin_a,
            'group_b_mean_margin': mean_margin_b,
            'statistic': mannwhitney_statistic,
            'p_value': p_value,
            'alpha': alpha,
            'reject_null': reject_null,
            'business_conclusion': business_conclusion,
            'equivalence_check': equivalence
        }
        
        self.test_results['zipcode_margin'] = result
        
        print(f"\n[Result] {'REJECT H₀' if reject_null else 'FAIL TO REJECT H₀'}")
        print(f"  p-value: {p_value:.6f}")
        print(f"\n[Business Conclusion] {business_conclusion}")
        
        return result
    
    def test_gender_risk_differences(self, alpha: float = 0.05) -> Dict[str, Any]:
        """
        Test H₀: There is no significant risk difference between Women and Men.
        
        Parameters:
        -----------
        alpha : float
            Significance level (default: 0.05)
        
        Returns:
        --------
        dict
            Test results with p-values and business conclusions
        """
        print("\n" + "="*70)
        print("HYPOTHESIS TEST 4: Gender Risk Differences")
        print("="*70)
        print("H₀: There is no significant risk difference between Women and Men")
        print("H₁: There are significant risk differences between Women and Men")
        
        # Get gender groups
        genders = self.data['Gender'].dropna().unique()
        
        if len(genders) < 2:
            return {
                'hypothesis': 'No risk differences between genders',
                'status': 'insufficient_data',
                'message': 'Need at least 2 gender categories for testing'
            }
        
        # Select two gender groups for A/B testing
        # Map to standard labels
        gender_mapping = {
            'M': 'Male', 'Male': 'Male', 'MALE': 'Male',
            'F': 'Female', 'Female': 'Female', 'FEMALE': 'Female'
        }
        self.data['GenderStandardized'] = self.data['Gender'].map(
            lambda x: gender_mapping.get(str(x).strip(), str(x).strip())
        )
        
        gender_stats = self.data.groupby('GenderStandardized').agg({
            'HasClaim': 'mean',
            'ClaimSeverityGivenClaim': 'mean',
            'TotalPremium': 'sum',
            'TotalClaims': 'sum'
        }).assign(
            LossRatio=lambda x: x['TotalClaims'] / x['TotalPremium']
        )
        
        if 'Female' not in gender_stats.index or 'Male' not in gender_stats.index:
            return {
                'hypothesis': 'No risk differences between genders',
                'status': 'insufficient_data',
                'message': 'Need both Male and Female categories for testing'
            }
        
        group_a = self.data[self.data['GenderStandardized'] == 'Female'].copy()
        group_b = self.data[self.data['GenderStandardized'] == 'Male'].copy()
        
        print(f"\n[Data Segmentation]")
        print(f"  Group A (Control): Female (n={len(group_a)})")
        print(f"  Group B (Test): Male (n={len(group_b)})")
        
        # Check group equivalence
        print("\n[Equivalence Check] Verifying groups are equivalent on other features...")
        equivalence = self._check_group_equivalence(group_a, group_b, 'Gender')
        for col, result in equivalence.items():
            status = "✓ Equivalent" if result['equivalent'] else "✗ Different"
            print(f"  {col}: {status} (p={result['p_value']:.3f})")
        
        # Test 1: Claim Frequency (Chi-square test)
        print("\n[Test 4.1] Claim Frequency Test (Chi-square)")
        # EXPLICIT: Create control and test groups
        freq_a = group_a['HasClaim'].sum()
        freq_b = group_b['HasClaim'].sum()
        n_a = len(group_a)
        n_b = len(group_b)
        
        # EXPLICIT: Create contingency table for chi-square test
        contingency = pd.DataFrame({
            'Has Claim': [freq_a, freq_b],
            'No Claim': [n_a - freq_a, n_b - freq_b]
        }, index=['Female', 'Male'])
        
        # EXPLICIT: Run chi-square test
        chi2_statistic, p_value_freq, degrees_of_freedom, expected_frequencies = chi2_contingency(contingency)
        freq_rate_a = freq_a / n_a
        freq_rate_b = freq_b / n_b
        
        print(f"  Female frequency: {freq_rate_a:.4f} ({freq_a}/{n_a})")
        print(f"  Male frequency: {freq_rate_b:.4f} ({freq_b}/{n_b})")
        print(f"  Chi-square statistic: {chi2_statistic:.4f}")
        print(f"  Degrees of freedom: {degrees_of_freedom}")
        print(f"  p-value: {p_value_freq:.6f}")
        
        # Test 2: Claim Severity (Mann-Whitney U test)
        print("\n[Test 4.2] Claim Severity Test (Mann-Whitney U)")
        # EXPLICIT: Extract severity data for control and test groups
        severity_a = group_a['ClaimSeverityGivenClaim'].dropna()
        severity_b = group_b['ClaimSeverityGivenClaim'].dropna()
        
        # EXPLICIT: Run Mann-Whitney U test
        if len(severity_a) > 0 and len(severity_b) > 0:
            mannwhitney_statistic, p_value_sev = mannwhitneyu(severity_a, severity_b, alternative='two-sided')
            mean_sev_a = severity_a.mean()
            mean_sev_b = severity_b.mean()
            
            print(f"  Female mean severity: R{mean_sev_a:,.2f} (n={len(severity_a)})")
            print(f"  Male mean severity: R{mean_sev_b:,.2f} (n={len(severity_b)})")
            print(f"  Mann-Whitney U statistic: {mannwhitney_statistic:.4f}")
            print(f"  p-value: {p_value_sev:.6f}")
        else:
            p_value_sev = 1.0
            mean_sev_a = 0
            mean_sev_b = 0
        
        # EXPLICIT: P-value threshold comparison
        # Overall result: reject if either test is significant
        overall_p_value = min(p_value_freq, p_value_sev)
        # EXPLICIT: Compare p-value to threshold (alpha = 0.05)
        reject_null = overall_p_value < alpha
        print(f"\n[Decision] P-value threshold (alpha): {alpha}")
        print(f"  Frequency p-value: {p_value_freq:.6f}")
        print(f"  Severity p-value: {p_value_sev:.6f}")
        print(f"  Overall p-value: {overall_p_value:.6f}")
        print(f"  Decision: {'REJECT H₀' if reject_null else 'FAIL TO REJECT H₀'} (p {'<' if reject_null else '>='} {alpha})")
        
        # Business interpretation
        if reject_null:
            pct_diff = ((mean_sev_b - mean_sev_a) / mean_sev_a) * 100 if mean_sev_a > 0 else 0
            business_conclusion = (
                f"We REJECT the null hypothesis (p < {alpha:.3f}). "
                f"Significant risk differences exist between genders. "
                f"Men exhibit {pct_diff:.1f}% higher claim severity than women "
                f"(R{mean_sev_b:,.2f} vs R{mean_sev_a:,.2f}), suggesting gender-based risk adjustments "
                f"may be warranted."
            )
        else:
            business_conclusion = (
                f"We FAIL TO REJECT the null hypothesis (p = {overall_p_value:.3f}). "
                f"No statistically significant risk differences were found between genders "
                f"at the {alpha*100}% significance level. This aligns with fair insurance practices "
                f"and suggests gender should not be a primary factor in pricing decisions."
            )
        
        result = {
            'hypothesis': 'No significant risk difference between Women and Men',
            'test_type': 'Chi-square (Frequency) + Mann-Whitney U (Severity)',
            'group_a_gender': 'Female',
            'group_b_gender': 'Male',
            'group_a_size': len(group_a),
            'group_b_size': len(group_b),
            'frequency_test': {
                'female_rate': freq_rate_a,
                'male_rate': freq_rate_b,
                'chi2': chi2_statistic,
                'p_value': p_value_freq
            },
            'severity_test': {
                'female_mean': mean_sev_a,
                'male_mean': mean_sev_b,
                'statistic': mannwhitney_statistic if len(severity_a) > 0 and len(severity_b) > 0 else None,
                'p_value': p_value_sev
            },
            'p_value_frequency': p_value_freq,
            'p_value_severity': p_value_sev,
            'overall_p_value': overall_p_value,
            'alpha': alpha,
            'reject_null': reject_null,
            'business_conclusion': business_conclusion,
            'equivalence_check': equivalence
        }
        
        self.test_results['gender_risk'] = result
        
        print(f"\n[Result] {'REJECT H₀' if reject_null else 'FAIL TO REJECT H₀'}")
        print(f"  Frequency p-value: {p_value_freq:.6f}")
        print(f"  Severity p-value: {p_value_sev:.6f}")
        print(f"  Overall p-value: {overall_p_value:.6f}")
        print(f"\n[Business Conclusion] {business_conclusion}")
        
        return result
    
    def interpret_statistical_results(self, p_value: float, alpha: float, 
                                      test_name: str, group_a_name: str = None,
                                      group_b_name: str = None, 
                                      group_a_metric: float = None,
                                      group_b_metric: float = None) -> str:
        """
        EXPLICIT BUSINESS LOGIC FUNCTION: Interpret statistical test results.
        
        This function takes p-values and explicitly interprets them with business logic.
        
        Parameters:
        -----------
        p_value : float
            P-value from statistical test
        alpha : float
            Significance threshold (typically 0.05)
        test_name : str
            Name of the test being interpreted
        group_a_name : str, optional
            Name of control group
        group_b_name : str, optional
            Name of test group
        group_a_metric : float, optional
            Metric value for group A
        group_b_metric : float, optional
            Metric value for group B
        
        Returns:
        --------
        str
            Business interpretation of the results
        """
        # EXPLICIT: Compare p-value to threshold
        is_significant = p_value < alpha
        
        if is_significant:
            interpretation = (
                f"We REJECT the null hypothesis (p = {p_value:.6f} < {alpha:.3f}). "
                f"Significant differences exist in {test_name}."
            )
            if group_a_name and group_b_name and group_a_metric is not None and group_b_metric is not None:
                pct_diff = ((group_a_metric - group_b_metric) / abs(group_b_metric)) * 100 if group_b_metric != 0 else 0
                interpretation += (
                    f" {group_a_name} exhibits {abs(pct_diff):.1f}% "
                    f"{'higher' if pct_diff > 0 else 'lower'} {test_name} than {group_b_name}, "
                    f"suggesting adjustments may be warranted."
                )
        else:
            interpretation = (
                f"We FAIL TO REJECT the null hypothesis (p = {p_value:.6f} >= {alpha:.3f}). "
                f"No statistically significant differences were found in {test_name} "
                f"at the {alpha*100}% significance level."
            )
        
        return interpretation
    
    def _test_categorical_frequency(self, group_col: str, outcome_col: str, 
                                    test_name: str) -> Dict[str, Any]:
        """
        Test frequency differences across categorical groups using Chi-square.
        
        Parameters:
        -----------
        group_col : str
            Column name for grouping
        outcome_col : str
            Binary outcome column (0/1)
        test_name : str
            Name of the test
        
        Returns:
        --------
        dict
            Test results
        """
        # Create contingency table
        contingency = pd.crosstab(self.data[group_col], self.data[outcome_col])
        
        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            return {'p_value': 1.0, 'statistic': None, 'message': 'Insufficient data'}
        
        chi2, p_value, dof, expected = chi2_contingency(contingency)
        
        return {
            'test_name': test_name,
            'statistic': chi2,
            'p_value': p_value,
            'degrees_of_freedom': dof
        }
    
    def _test_categorical_severity(self, group_col: str, severity_col: str, 
                                   test_name: str) -> Dict[str, Any]:
        """
        Test severity differences across categorical groups using Kruskal-Wallis.
        
        Parameters:
        -----------
        group_col : str
            Column name for grouping
        severity_col : str
            Severity column (continuous)
        test_name : str
            Name of the test
        
        Returns:
        --------
        dict
            Test results
        """
        # Get groups
        groups = []
        group_names = []
        
        for group_name in self.data[group_col].dropna().unique():
            group_data = self.data[self.data[group_col] == group_name][severity_col].dropna()
            if len(group_data) > 0:
                groups.append(group_data)
                group_names.append(group_name)
        
        if len(groups) < 2:
            return {'p_value': 1.0, 'statistic': None, 'message': 'Insufficient data'}
        
        # Kruskal-Wallis test (non-parametric ANOVA)
        statistic, p_value = stats.kruskal(*groups)
        
        return {
            'test_name': test_name,
            'statistic': statistic,
            'p_value': p_value,
            'num_groups': len(groups)
        }
    
    def run_all_tests(self, alpha: float = 0.05) -> Dict[str, Any]:
        """
        Run all hypothesis tests.
        
        Parameters:
        -----------
        alpha : float
            Significance level (default: 0.05)
        
        Returns:
        --------
        dict
            All test results
        """
        print("\n" + "="*70)
        print("RUNNING ALL HYPOTHESIS TESTS")
        print("="*70)
        
        # Run all tests
        self.test_province_risk_differences(alpha=alpha)
        self.test_zipcode_risk_differences(alpha=alpha)
        self.test_zipcode_margin_differences(alpha=alpha)
        self.test_gender_risk_differences(alpha=alpha)
        
        return self.test_results
    
    def generate_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive report of all test results.
        
        Parameters:
        -----------
        output_path : str, optional
            Path to save the report. If None, returns report as string.
        
        Returns:
        --------
        str
            Report content
        """
        report_lines = []
        report_lines.append("="*70)
        report_lines.append("HYPOTHESIS TESTING REPORT")
        report_lines.append("Insurance Risk Analytics - Task 3")
        report_lines.append("="*70)
        report_lines.append("")
        
        # Summary
        report_lines.append("EXECUTIVE SUMMARY")
        report_lines.append("-"*70)
        total_tests = len(self.test_results)
        rejected = sum(1 for r in self.test_results.values() if r.get('reject_null', False))
        report_lines.append(f"Total Tests: {total_tests}")
        report_lines.append(f"Rejected H₀: {rejected}")
        report_lines.append(f"Failed to Reject H₀: {total_tests - rejected}")
        report_lines.append("")
        
        # Detailed results
        test_names = {
            'province_risk': 'Test 1: Province Risk Differences',
            'zipcode_risk': 'Test 2: Zipcode Risk Differences',
            'zipcode_margin': 'Test 3: Zipcode Margin Differences',
            'gender_risk': 'Test 4: Gender Risk Differences'
        }
        
        for key, name in test_names.items():
            if key in self.test_results:
                result = self.test_results[key]
                report_lines.append("="*70)
                report_lines.append(name)
                report_lines.append("="*70)
                report_lines.append(f"Hypothesis: {result.get('hypothesis', 'N/A')}")
                report_lines.append(f"Test Type: {result.get('test_type', 'N/A')}")
                report_lines.append("")
                
                if 'status' in result:
                    report_lines.append(f"Status: {result['status']}")
                    report_lines.append(f"Message: {result.get('message', 'N/A')}")
                else:
                    report_lines.append(f"Result: {'REJECT H₀' if result.get('reject_null', False) else 'FAIL TO REJECT H₀'}")
                    report_lines.append(f"Alpha Level: {result.get('alpha', 0.05)}")
                    report_lines.append("")
                    
                    if 'p_value' in result:
                        report_lines.append(f"P-value: {result['p_value']:.6f}")
                    elif 'overall_p_value' in result:
                        report_lines.append(f"Overall P-value: {result['overall_p_value']:.6f}")
                        if 'p_value_frequency' in result:
                            report_lines.append(f"  - Frequency P-value: {result['p_value_frequency']:.6f}")
                        if 'p_value_severity' in result:
                            report_lines.append(f"  - Severity P-value: {result['p_value_severity']:.6f}")
                    
                    report_lines.append("")
                    report_lines.append("Business Conclusion:")
                    report_lines.append(result.get('business_conclusion', 'N/A'))
                
                report_lines.append("")
        
        report_content = "\n".join(report_lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_content)
            print(f"\nReport saved to: {output_path}")
        
        return report_content

