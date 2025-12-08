"""
Hypothesis Testing Module for Insurance Risk Analytics.
Object-oriented implementation for A/B testing and statistical validation.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
import warnings

warnings.filterwarnings("ignore")


class HypothesisTester:
    """Class for conducting statistical hypothesis tests."""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize HypothesisTester.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input dataframe with insurance data
        """
        self.data = data.copy()
        self.results: Dict[str, Any] = {}
    
    def calculate_claim_frequency(self, group_col: str = None) -> pd.Series:
        """
        Calculate Claim Frequency (proportion of policies with at least one claim).
        
        Parameters:
        -----------
        group_col : str, optional
            Column to group by. If None, returns overall frequency.
        
        Returns:
        --------
        pd.Series
            Claim frequency by group
        """
        # Create binary claim indicator
        self.data['HasClaim'] = (self.data['TotalClaims'] > 0).astype(int)
        
        if group_col is None:
            return self.data['HasClaim'].mean()
        
        if group_col not in self.data.columns:
            raise ValueError(f"Column '{group_col}' not found in data")
        
        # Calculate frequency by group
        frequency = self.data.groupby(group_col)['HasClaim'].agg(['sum', 'count'])
        frequency['ClaimFrequency'] = frequency['sum'] / frequency['count']
        
        return frequency['ClaimFrequency']
    
    def calculate_claim_severity(self, group_col: str = None) -> pd.Series:
        """
        Calculate Claim Severity (average amount of a claim, given a claim occurred).
        
        Parameters:
        -----------
        group_col : str, optional
            Column to group by. If None, returns overall severity.
        
        Returns:
        --------
        pd.Series
            Claim severity by group
        """
        # Filter to only policies with claims
        claims_data = self.data[self.data['TotalClaims'] > 0].copy()
        
        if len(claims_data) == 0:
            return pd.Series(dtype=float)
        
        if group_col is None:
            return claims_data['TotalClaims'].mean()
        
        if group_col not in self.data.columns:
            raise ValueError(f"Column '{group_col}' not found in data")
        
        severity = claims_data.groupby(group_col)['TotalClaims'].mean()
        return severity
    
    def calculate_margin(self, group_col: str = None) -> pd.Series:
        """
        Calculate Margin (TotalPremium - TotalClaims).
        
        Parameters:
        -----------
        group_col : str, optional
            Column to group by. If None, returns overall margin.
        
        Returns:
        --------
        pd.Series
            Margin by group
        """
        self.data['Margin'] = self.data['TotalPremium'] - self.data['TotalClaims']
        
        if group_col is None:
            return self.data['Margin'].mean()
        
        if group_col not in self.data.columns:
            raise ValueError(f"Column '{group_col}' not found in data")
        
        margin = self.data.groupby(group_col)['Margin'].mean()
        return margin
    
    def chi_square_test(self, group_col: str, metric_col: str = 'HasClaim') -> Dict[str, Any]:
        """
        Perform Chi-square test for independence.
        
        Parameters:
        -----------
        group_col : str
            Column to group by
        metric_col : str
            Binary metric column (default: 'HasClaim')
        
        Returns:
        --------
        dict
            Test results with statistic, p-value, and interpretation
        """
        if metric_col not in self.data.columns:
            if metric_col == 'HasClaim':
                self.data['HasClaim'] = (self.data['TotalClaims'] > 0).astype(int)
            else:
                raise ValueError(f"Column '{metric_col}' not found")
        
        # Create contingency table
        contingency = pd.crosstab(self.data[group_col], self.data[metric_col])
        
        # Perform chi-square test
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
        
        result = {
            'test_type': 'Chi-square',
            'null_hypothesis': f'No association between {group_col} and {metric_col}',
            'chi2_statistic': chi2,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'contingency_table': contingency,
            'reject_null': p_value < 0.05,
            'interpretation': 'Reject H₀' if p_value < 0.05 else 'Fail to reject H₀'
        }
        
        return result
    
    def t_test(self, group_col: str, value_col: str, group_a: str = None, group_b: str = None) -> Dict[str, Any]:
        """
        Perform independent samples t-test.
        
        Parameters:
        -----------
        group_col : str
            Column to group by
        value_col : str
            Numeric column to test
        group_a : str, optional
            First group value. If None, uses first unique value.
        group_b : str, optional
            Second group value. If None, uses second unique value.
        
        Returns:
        --------
        dict
            Test results with statistic, p-value, and interpretation
        """
        if value_col not in self.data.columns:
            raise ValueError(f"Column '{value_col}' not found in data")
        
        unique_groups = self.data[group_col].dropna().unique()
        
        if len(unique_groups) < 2:
            raise ValueError(f"Need at least 2 groups in {group_col}")
        
        if group_a is None:
            group_a = unique_groups[0]
        if group_b is None:
            group_b = unique_groups[1]
        
        # Get data for each group
        group_a_data = self.data[self.data[group_col] == group_a][value_col].dropna()
        group_b_data = self.data[self.data[group_col] == group_b][value_col].dropna()
        
        if len(group_a_data) == 0 or len(group_b_data) == 0:
            raise ValueError("One or both groups have no data")
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(group_a_data, group_b_data)
        
        # Calculate means
        mean_a = group_a_data.mean()
        mean_b = group_b_data.mean()
        mean_diff = mean_a - mean_b
        percent_diff = (mean_diff / mean_b * 100) if mean_b != 0 else 0
        
        result = {
            'test_type': 'Independent Samples t-test',
            'null_hypothesis': f'No difference in {value_col} between {group_a} and {group_b}',
            'group_a': group_a,
            'group_b': group_b,
            'group_a_mean': mean_a,
            'group_b_mean': mean_b,
            'mean_difference': mean_diff,
            'percent_difference': percent_diff,
            't_statistic': t_stat,
            'p_value': p_value,
            'group_a_n': len(group_a_data),
            'group_b_n': len(group_b_data),
            'reject_null': p_value < 0.05,
            'interpretation': 'Reject H₀' if p_value < 0.05 else 'Fail to reject H₀'
        }
        
        return result
    
    def anova_test(self, group_col: str, value_col: str) -> Dict[str, Any]:
        """
        Perform one-way ANOVA test.
        
        Parameters:
        -----------
        group_col : str
            Column to group by
        value_col : str
            Numeric column to test
        
        Returns:
        --------
        dict
            Test results with F-statistic, p-value, and interpretation
        """
        if value_col not in self.data.columns:
            raise ValueError(f"Column '{value_col}' not found in data")
        
        # Get groups
        groups = []
        group_names = []
        
        for group_name in self.data[group_col].dropna().unique():
            group_data = self.data[self.data[group_col] == group_name][value_col].dropna()
            if len(group_data) > 0:
                groups.append(group_data)
                group_names.append(group_name)
        
        if len(groups) < 2:
            raise ValueError(f"Need at least 2 groups in {group_col}")
        
        # Perform ANOVA
        f_stat, p_value = stats.f_oneway(*groups)
        
        # Calculate group means
        group_means = {name: data.mean() for name, data in zip(group_names, groups)}
        group_counts = {name: len(data) for name, data in zip(group_names, groups)}
        
        result = {
            'test_type': 'One-way ANOVA',
            'null_hypothesis': f'No difference in {value_col} across {group_col}',
            'f_statistic': f_stat,
            'p_value': p_value,
            'group_means': group_means,
            'group_counts': group_counts,
            'num_groups': len(groups),
            'reject_null': p_value < 0.05,
            'interpretation': 'Reject H₀' if p_value < 0.05 else 'Fail to reject H₀'
        }
        
        return result
    
    def test_province_risk_differences(self) -> Dict[str, Any]:
        """
        Test H₀: There are no risk differences across provinces.
        
        Tests both Claim Frequency and Claim Severity.
        
        Returns:
        --------
        dict
            Test results for both metrics
        """
        results = {}
        
        # Test Claim Frequency using Chi-square
        try:
            freq_result = self.chi_square_test('Province', 'HasClaim')
            results['claim_frequency'] = freq_result
        except Exception as e:
            results['claim_frequency'] = {'error': str(e)}
        
        # Test Claim Severity using ANOVA
        try:
            severity_result = self.anova_test('Province', 'TotalClaims')
            results['claim_severity'] = severity_result
        except Exception as e:
            results['claim_severity'] = {'error': str(e)}
        
        # Calculate summary statistics
        freq_by_province = self.calculate_claim_frequency('Province')
        severity_by_province = self.calculate_claim_severity('Province')
        
        results['summary'] = {
            'claim_frequency_by_province': freq_by_province.to_dict(),
            'claim_severity_by_province': severity_by_province.to_dict()
        }
        
        self.results['province_risk'] = results
        return results
    
    def test_zipcode_risk_differences(self, top_n: int = 10) -> Dict[str, Any]:
        """
        Test H₀: There are no risk differences between zip codes.
        
        Tests top N zip codes by policy count.
        
        Parameters:
        -----------
        top_n : int
            Number of top zip codes to test
        
        Returns:
        --------
        dict
            Test results
        """
        # Get top N zip codes by policy count
        zipcode_counts = self.data['PostalCode'].value_counts().head(top_n)
        top_zipcodes = zipcode_counts.index.tolist()
        
        # Filter data to top zip codes
        filtered_data = self.data[self.data['PostalCode'].isin(top_zipcodes)].copy()
        
        if len(filtered_data) == 0:
            return {'error': 'No data for top zip codes'}
        
        # Create temporary tester with filtered data
        temp_tester = HypothesisTester(filtered_data)
        results = {}
        
        # Test Claim Frequency using Chi-square
        try:
            freq_result = temp_tester.chi_square_test('PostalCode', 'HasClaim')
            results['claim_frequency'] = freq_result
        except Exception as e:
            results['claim_frequency'] = {'error': str(e)}
        
        # Test Claim Severity using ANOVA
        try:
            severity_result = temp_tester.anova_test('PostalCode', 'TotalClaims')
            results['claim_severity'] = severity_result
        except Exception as e:
            results['claim_severity'] = {'error': str(e)}
        
        # Calculate summary statistics
        freq_by_zipcode = temp_tester.calculate_claim_frequency('PostalCode')
        severity_by_zipcode = temp_tester.calculate_claim_severity('PostalCode')
        
        results['summary'] = {
            'top_zipcodes': top_zipcodes,
            'claim_frequency_by_zipcode': freq_by_zipcode.to_dict() if len(freq_by_zipcode) > 0 else {},
            'claim_severity_by_zipcode': severity_by_zipcode.to_dict() if len(severity_by_zipcode) > 0 else {}
        }
        
        self.results['zipcode_risk'] = results
        return results
    
    def test_zipcode_margin_differences(self, top_n: int = 10) -> Dict[str, Any]:
        """
        Test H₀: There is no significant margin (profit) difference between zip codes.
        
        Parameters:
        -----------
        top_n : int
            Number of top zip codes to test
        
        Returns:
        --------
        dict
            Test results
        """
        # Get top N zip codes by policy count
        zipcode_counts = self.data['PostalCode'].value_counts().head(top_n)
        top_zipcodes = zipcode_counts.index.tolist()
        
        # Filter data to top zip codes
        filtered_data = self.data[self.data['PostalCode'].isin(top_zipcodes)].copy()
        
        if len(filtered_data) == 0:
            return {'error': 'No data for top zip codes'}
        
        # Create temporary tester with filtered data
        temp_tester = HypothesisTester(filtered_data)
        results = {}
        
        # Test Margin using ANOVA
        try:
            margin_result = temp_tester.anova_test('PostalCode', 'Margin')
            results['margin'] = margin_result
        except Exception as e:
            results['margin'] = {'error': str(e)}
        
        # Calculate summary statistics
        margin_by_zipcode = temp_tester.calculate_margin('PostalCode')
        
        results['summary'] = {
            'top_zipcodes': top_zipcodes,
            'margin_by_zipcode': margin_by_zipcode.to_dict() if len(margin_by_zipcode) > 0 else {}
        }
        
        self.results['zipcode_margin'] = results
        return results
    
    def test_gender_risk_differences(self) -> Dict[str, Any]:
        """
        Test H₀: There is no significant risk difference between Women and Men.
        
        Returns:
        --------
        dict
            Test results
        """
        # Filter to only Male and Female (exclude "Not specified")
        gender_data = self.data[self.data['Gender'].isin(['Male', 'Female'])].copy()
        
        if len(gender_data) == 0:
            return {'error': 'No data for Male/Female comparison'}
        
        # Create temporary tester with filtered data
        temp_tester = HypothesisTester(gender_data)
        results = {}
        
        # Test Claim Frequency using Chi-square
        try:
            freq_result = temp_tester.chi_square_test('Gender', 'HasClaim')
            results['claim_frequency'] = freq_result
        except Exception as e:
            results['claim_frequency'] = {'error': str(e)}
        
        # Test Claim Severity using t-test
        try:
            severity_result = temp_tester.t_test('Gender', 'TotalClaims', 'Male', 'Female')
            results['claim_severity'] = severity_result
        except Exception as e:
            results['claim_severity'] = {'error': str(e)}
        
        # Calculate summary statistics
        freq_by_gender = temp_tester.calculate_claim_frequency('Gender')
        severity_by_gender = temp_tester.calculate_claim_severity('Gender')
        margin_by_gender = temp_tester.calculate_margin('Gender')
        
        results['summary'] = {
            'claim_frequency_by_gender': freq_by_gender.to_dict(),
            'claim_severity_by_gender': severity_by_gender.to_dict(),
            'margin_by_gender': margin_by_gender.to_dict()
        }
        
        self.results['gender_risk'] = results
        return results
    
    def generate_report(self) -> str:
        """
        Generate a comprehensive report of all hypothesis tests.
        
        Returns:
        --------
        str
            Formatted report string
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("HYPOTHESIS TESTING REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        for test_name, test_results in self.results.items():
            report_lines.append(f"\n{'=' * 80}")
            report_lines.append(f"TEST: {test_name.upper().replace('_', ' ')}")
            report_lines.append(f"{'=' * 80}\n")
            
            if 'error' in test_results:
                report_lines.append(f"Error: {test_results['error']}\n")
                continue
            
            # Process each metric
            for metric_name, metric_result in test_results.items():
                if metric_name == 'summary':
                    continue
                
                if 'error' in metric_result:
                    report_lines.append(f"{metric_name}: Error - {metric_result['error']}\n")
                    continue
                
                report_lines.append(f"\n{metric_name.upper().replace('_', ' ')}:")
                report_lines.append(f"  Test Type: {metric_result.get('test_type', 'N/A')}")
                report_lines.append(f"  Null Hypothesis: {metric_result.get('null_hypothesis', 'N/A')}")
                
                if 'p_value' in metric_result:
                    report_lines.append(f"  P-value: {metric_result['p_value']:.6f}")
                    report_lines.append(f"  Significance Level: α = 0.05")
                    report_lines.append(f"  Result: {metric_result['interpretation']}")
                    
                    if metric_result['reject_null']:
                        report_lines.append(f"  ⚠️  REJECT H₀: Statistically significant difference detected")
                    else:
                        report_lines.append(f"  ✓  FAIL TO REJECT H₀: No statistically significant difference")
                
                if 'mean_difference' in metric_result:
                    report_lines.append(f"  Group A ({metric_result.get('group_a', 'N/A')}): Mean = {metric_result.get('group_a_mean', 0):.2f}")
                    report_lines.append(f"  Group B ({metric_result.get('group_b', 'N/A')}): Mean = {metric_result.get('group_b_mean', 0):.2f}")
                    report_lines.append(f"  Difference: {metric_result.get('mean_difference', 0):.2f} ({metric_result.get('percent_difference', 0):.2f}%)")
                
                report_lines.append("")
        
        return "\n".join(report_lines)

