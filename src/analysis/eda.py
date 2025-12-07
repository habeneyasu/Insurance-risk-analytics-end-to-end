"""
Exploratory Data Analysis (EDA) for insurance risk analytics.
Object-oriented implementation.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
import warnings

warnings.filterwarnings("ignore")


class EDAAnalyzer:
    """Class for performing exploratory data analysis."""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize EDAAnalyzer.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input dataframe
        """
        self.data = data.copy()
        self.loss_ratio_results: Optional[Dict[str, Any]] = None
        self.distribution_stats: Optional[Dict[str, Any]] = None
        self.temporal_trends: Optional[pd.DataFrame] = None
        self.vehicle_risk: Optional[pd.DataFrame] = None
    
    def calculate_loss_ratio(self) -> Dict[str, Any]:
        """
        Calculate Loss Ratio (TotalClaims / TotalPremium) for the portfolio.
        
        Returns:
        --------
        dict
            Summary of loss ratios
        """
        # Overall loss ratio
        total_claims = self.data["TotalClaims"].sum()
        total_premium = self.data["TotalPremium"].sum()
        overall_loss_ratio = total_claims / total_premium if total_premium > 0 else 0
        
        # Loss ratio by Province
        loss_ratio_by_province = (
            self.data.groupby("Province")
            .agg({"TotalClaims": "sum", "TotalPremium": "sum"})
            .assign(LossRatio=lambda x: x["TotalClaims"] / x["TotalPremium"])
            .sort_values("LossRatio", ascending=False)
        )
        
        # Loss ratio by VehicleType
        loss_ratio_by_vehicle = (
            self.data.groupby("VehicleType")
            .agg({"TotalClaims": "sum", "TotalPremium": "sum"})
            .assign(LossRatio=lambda x: x["TotalClaims"] / x["TotalPremium"])
            .sort_values("LossRatio", ascending=False)
        )
        
        # Loss ratio by Gender
        loss_ratio_by_gender = (
            self.data.groupby("Gender")
            .agg({"TotalClaims": "sum", "TotalPremium": "sum"})
            .assign(LossRatio=lambda x: x["TotalClaims"] / x["TotalPremium"])
            .sort_values("LossRatio", ascending=False)
        )
        
        self.loss_ratio_results = {
            "overall_loss_ratio": overall_loss_ratio,
            "by_province": loss_ratio_by_province,
            "by_vehicle_type": loss_ratio_by_vehicle,
            "by_gender": loss_ratio_by_gender,
        }
        
        return self.loss_ratio_results
    
    def analyze_distributions(self, numerical_cols: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze distributions of numerical variables.
        
        Parameters:
        -----------
        numerical_cols : List[str], optional
            List of numerical column names to analyze. If None, analyzes all numerical columns.
        
        Returns:
        --------
        dict
            Dictionary containing distribution statistics
        """
        if numerical_cols is None:
            numerical_cols = list(self.data.select_dtypes(include=[np.number]).columns)
        
        distributions = {}
        
        for col in numerical_cols:
            if col in self.data.columns:
                col_data = self.data[col].dropna()
                if len(col_data) > 0:
                    distributions[col] = {
                        "mean": col_data.mean(),
                        "median": col_data.median(),
                        "std": col_data.std(),
                        "min": col_data.min(),
                        "max": col_data.max(),
                        "q25": col_data.quantile(0.25),
                        "q75": col_data.quantile(0.75),
                        "skewness": col_data.skew(),
                        "kurtosis": col_data.kurtosis(),
                        "count": len(col_data),
                    }
        
        self.distribution_stats = distributions
        return distributions
    
    def analyze_temporal_trends(self, date_col: str = "TransactionMonth") -> pd.DataFrame:
        """
        Analyze temporal trends in claims and premiums.
        
        Parameters:
        -----------
        date_col : str
            Name of the date column
        
        Returns:
        --------
        pd.DataFrame
            Monthly aggregated statistics
        """
        # Convert to datetime if needed
        if self.data[date_col].dtype == "object":
            self.data[date_col] = pd.to_datetime(self.data[date_col], errors="coerce")
        
        monthly_stats = (
            self.data.groupby(self.data[date_col].dt.to_period("M"))
            .agg(
                {
                    "TotalClaims": ["sum", "mean", "count"],
                    "TotalPremium": ["sum", "mean"],
                }
            )
            .reset_index()
        )
        
        monthly_stats.columns = [
            "Month",
            "TotalClaims_Sum",
            "TotalClaims_Mean",
            "ClaimCount",
            "TotalPremium_Sum",
            "TotalPremium_Mean",
        ]
        
        monthly_stats["ClaimFrequency"] = monthly_stats["ClaimCount"] / len(self.data) * 100
        monthly_stats["LossRatio"] = (
            monthly_stats["TotalClaims_Sum"] / monthly_stats["TotalPremium_Sum"]
        )
        
        self.temporal_trends = monthly_stats
        return monthly_stats
    
    def analyze_vehicle_risk(self) -> pd.DataFrame:
        """
        Analyze vehicle make/model risk (highest and lowest claim amounts).
        
        Returns:
        --------
        pd.DataFrame
            Vehicle risk analysis
        """
        # Find make and model columns (case-insensitive)
        make_col = next((col for col in self.data.columns if col.lower() == "make"), None)
        model_col = next((col for col in self.data.columns if col.lower() == "model"), None)
        
        if not make_col or not model_col:
            raise ValueError("Make and Model columns not found in data")
        
        groupby_cols = [make_col, model_col]
        
        vehicle_risk = (
            self.data.groupby(groupby_cols)
            .agg(
                {
                    "TotalClaims": ["sum", "mean", "count"],
                    "TotalPremium": "sum",
                    "CustomValueEstimate": "mean",
                }
            )
            .reset_index()
        )
        
        # Use original column names in output
        vehicle_risk.columns = [
            make_col,
            model_col,
            "TotalClaims_Sum",
            "TotalClaims_Mean",
            "PolicyCount",
            "TotalPremium_Sum",
            "AvgCustomValue",
        ]
        
        # Rename for consistency in downstream code
        vehicle_risk = vehicle_risk.rename(columns={make_col: "Make", model_col: "Model"})
        
        vehicle_risk["LossRatio"] = (
            vehicle_risk["TotalClaims_Sum"] / vehicle_risk["TotalPremium_Sum"]
        )
        vehicle_risk["AvgClaimPerPolicy"] = (
            vehicle_risk["TotalClaims_Sum"] / vehicle_risk["PolicyCount"]
        )
        
        # Sort by total claims
        vehicle_risk = vehicle_risk.sort_values("TotalClaims_Sum", ascending=False)
        
        self.vehicle_risk = vehicle_risk
        return vehicle_risk
    
    def get_descriptive_statistics(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get descriptive statistics for numerical columns.
        
        Parameters:
        -----------
        columns : List[str], optional
            Columns to analyze. If None, analyzes all numerical columns.
        
        Returns:
        --------
        pd.DataFrame
            Descriptive statistics
        """
        if columns is None:
            columns = list(self.data.select_dtypes(include=[np.number]).columns)
        
        available_columns = [col for col in columns if col in self.data.columns]
        
        if not available_columns:
            return pd.DataFrame()
        
        return self.data[available_columns].describe()
    
    def analyze_geographic_trends(self) -> pd.DataFrame:
        """
        Analyze trends by geographic location (Province, PostalCode).
        
        Returns:
        --------
        pd.DataFrame
            Geographic analysis
        """
        geo_analysis = (
            self.data.groupby(["Province", "PostalCode"])
            .agg(
                {
                    "TotalClaims": ["sum", "mean", "count"],
                    "TotalPremium": "sum",
                    "PolicyID": "nunique",
                }
            )
            .reset_index()
        )
        
        geo_analysis.columns = [
            "Province",
            "PostalCode",
            "TotalClaims_Sum",
            "TotalClaims_Mean",
            "ClaimCount",
            "TotalPremium_Sum",
            "UniquePolicies",
        ]
        
        geo_analysis["LossRatio"] = (
            geo_analysis["TotalClaims_Sum"] / geo_analysis["TotalPremium_Sum"]
        )
        
        return geo_analysis.sort_values("LossRatio", ascending=False)
    
    def print_loss_ratio_summary(self) -> None:
        """Print loss ratio analysis summary."""
        if self.loss_ratio_results is None:
            self.calculate_loss_ratio()
        
        print("\n" + "="*60)
        print("LOSS RATIO ANALYSIS")
        print("="*60)
        print(f"Overall Loss Ratio: {self.loss_ratio_results['overall_loss_ratio']:.4f}")
        
        print("\nTop 5 Provinces by Loss Ratio:")
        top_provinces = self.loss_ratio_results["by_province"].head(5)
        for province, row in top_provinces.iterrows():
            print(f"  {province}: {row['LossRatio']:.4f}")
        
        print("\nTop 5 Vehicle Types by Loss Ratio:")
        top_vehicles = self.loss_ratio_results["by_vehicle_type"].head(5)
        for vehicle, row in top_vehicles.iterrows():
            print(f"  {vehicle}: {row['LossRatio']:.4f}")
        
        print("\nLoss Ratio by Gender:")
        for gender, row in self.loss_ratio_results["by_gender"].iterrows():
            print(f"  {gender}: {row['LossRatio']:.4f}")
        
        print("="*60 + "\n")
