"""
Data quality assessment module.
Object-oriented implementation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any


class DataQualityChecker:
    """Class for assessing data quality."""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize DataQualityChecker.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input dataframe
        """
        self.data = data
        self.quality_report: Dict[str, Any] = {}
    
    def check_missing_values(self) -> Dict[str, Any]:
        """
        Check for missing values in the dataset.
        
        Returns:
        --------
        dict
            Missing values report
        """
        missing_count = self.data.isnull().sum()
        missing_percentage = (missing_count / len(self.data)) * 100
        
        report = {
            "total_missing": missing_count.sum(),
            "columns_with_missing": missing_count[missing_count > 0].to_dict(),
            "missing_percentage": missing_percentage[missing_percentage > 0].to_dict(),
        }
        
        self.quality_report["missing_values"] = report
        return report
    
    def check_data_types(self) -> Dict[str, Any]:
        """
        Check data types and identify potential issues.
        
        Returns:
        --------
        dict
            Data types report
        """
        dtypes = self.data.dtypes.to_dict()
        numerical_cols = list(self.data.select_dtypes(include=[np.number]).columns)
        categorical_cols = list(self.data.select_dtypes(include=["object", "bool"]).columns)
        
        report = {
            "dtypes": dtypes,
            "numerical_columns": numerical_cols,
            "categorical_columns": categorical_cols,
            "total_columns": len(self.data.columns),
        }
        
        self.quality_report["data_types"] = report
        return report
    
    def check_duplicates(self) -> Dict[str, Any]:
        """
        Check for duplicate rows.
        
        Returns:
        --------
        dict
            Duplicates report
        """
        duplicate_count = self.data.duplicated().sum()
        duplicate_percentage = (duplicate_count / len(self.data)) * 100
        
        report = {
            "duplicate_rows": duplicate_count,
            "duplicate_percentage": duplicate_percentage,
            "unique_rows": len(self.data) - duplicate_count,
        }
        
        self.quality_report["duplicates"] = report
        return report
    
    def check_outliers(self, columns: List[str] = None, method: str = "iqr") -> Dict[str, Any]:
        """
        Check for outliers in specified columns.
        
        Parameters:
        -----------
        columns : List[str], optional
            Columns to check. If None, checks all numerical columns.
        method : str
            Method for outlier detection ('iqr' or 'zscore')
        
        Returns:
        --------
        dict
            Outliers report
        """
        if columns is None:
            columns = list(self.data.select_dtypes(include=[np.number]).columns)
        
        outliers_report = {}
        
        for col in columns:
            if col not in self.data.columns:
                continue
            
            if method == "iqr":
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = ((self.data[col] < lower_bound) | (self.data[col] > upper_bound)).sum()
            elif method == "zscore":
                z_scores = np.abs((self.data[col] - self.data[col].mean()) / self.data[col].std())
                outliers = (z_scores > 3).sum()
            else:
                raise ValueError("Method must be 'iqr' or 'zscore'")
            
            outliers_report[col] = {
                "outlier_count": outliers,
                "outlier_percentage": (outliers / len(self.data)) * 100,
            }
        
        self.quality_report["outliers"] = outliers_report
        return outliers_report
    
    def generate_quality_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive data quality report.
        
        Returns:
        --------
        dict
            Complete quality report
        """
        self.check_missing_values()
        self.check_data_types()
        self.check_duplicates()
        
        # Check outliers for key numerical columns
        key_columns = ["TotalPremium", "TotalClaims", "CustomValueEstimate"]
        available_columns = [col for col in key_columns if col in self.data.columns]
        if available_columns:
            self.check_outliers(columns=available_columns)
        
        return self.quality_report
    
    def print_quality_report(self) -> None:
        """Print quality report to console."""
        if not self.quality_report:
            self.generate_quality_report()
        
        print("\n" + "="*60)
        print("DATA QUALITY REPORT")
        print("="*60)
        
        # Missing values
        if "missing_values" in self.quality_report:
            missing = self.quality_report["missing_values"]
            print(f"\nMissing Values:")
            print(f"  Total missing: {missing['total_missing']}")
            if missing["columns_with_missing"]:
                print("  Top columns with missing values:")
                for col, count in list(missing["columns_with_missing"].items())[:5]:
                    pct = missing["missing_percentage"].get(col, 0)
                    print(f"    {col}: {count} ({pct:.2f}%)")
        
        # Duplicates
        if "duplicates" in self.quality_report:
            dup = self.quality_report["duplicates"]
            print(f"\nDuplicates: {dup['duplicate_rows']} ({dup['duplicate_percentage']:.2f}%)")
        
        # Outliers
        if "outliers" in self.quality_report:
            print(f"\nOutliers (IQR method):")
            for col, info in list(self.quality_report["outliers"].items())[:5]:
                print(f"  {col}: {info['outlier_count']} ({info['outlier_percentage']:.2f}%)")
        
        print("="*60 + "\n")

