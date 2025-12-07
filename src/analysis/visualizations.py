"""
Visualization generation for EDA.
Object-oriented implementation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List
import warnings

warnings.filterwarnings("ignore")

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 10


class VisualizationGenerator:
    """Class for generating visualizations."""
    
    def __init__(self, data: pd.DataFrame, output_dir: str = "reports/figures"):
        """
        Initialize VisualizationGenerator.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input dataframe
        output_dir : str
            Directory to save figures
        """
        self.data = data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_loss_ratio_by_province(self, loss_ratio_data: pd.DataFrame, save: bool = True) -> None:
        """
        Plot loss ratio by province.
        
        Parameters:
        -----------
        loss_ratio_data : pd.DataFrame
            Loss ratio data by province
        save : bool
            Whether to save the figure
        """
        fig, ax = plt.subplots(figsize=(14, 8))
        loss_ratio_data["LossRatio"].plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
        ax.set_title("Loss Ratio by Province", fontsize=16, fontweight="bold", pad=20)
        ax.set_xlabel("Province", fontsize=14, fontweight="bold")
        ax.set_ylabel("Loss Ratio (TotalClaims / TotalPremium)", fontsize=14, fontweight="bold")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / "loss_ratio_by_province.png", dpi=300, bbox_inches="tight")
            print(f"Saved: {self.output_dir / 'loss_ratio_by_province.png'}")
        plt.close()
    
    def plot_claims_distribution(self, save: bool = True) -> None:
        """
        Plot distribution of TotalClaims with outlier detection.
        
        Parameters:
        -----------
        save : bool
            Whether to save the figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        claims_data = self.data["TotalClaims"].dropna()
        
        # Histogram
        axes[0].hist(claims_data, bins=50, edgecolor="black", alpha=0.7, color="skyblue")
        axes[0].set_title("Distribution of Total Claims", fontsize=14, fontweight="bold")
        axes[0].set_xlabel("Total Claims", fontsize=12)
        axes[0].set_ylabel("Frequency", fontsize=12)
        axes[0].grid(axis="y", alpha=0.3)
        
        # Box plot
        bp = axes[1].boxplot(claims_data, vert=True, patch_artist=True)
        bp["boxes"][0].set_facecolor("lightcoral")
        axes[1].set_title("Total Claims - Box Plot (Outlier Detection)", fontsize=14, fontweight="bold")
        axes[1].set_ylabel("Total Claims", fontsize=12)
        axes[1].grid(axis="y", alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / "total_claims_distribution.png", dpi=300, bbox_inches="tight")
            print(f"Saved: {self.output_dir / 'total_claims_distribution.png'}")
        plt.close()
    
    def plot_temporal_trends(self, monthly_stats: pd.DataFrame, save: bool = True) -> None:
        """
        Plot temporal trends in claims and loss ratio.
        
        Parameters:
        -----------
        monthly_stats : pd.DataFrame
            Monthly aggregated statistics
        save : bool
            Whether to save the figure
        """
        fig, axes = plt.subplots(2, 1, figsize=(16, 12))
        
        months = monthly_stats["Month"].astype(str)
        
        # Monthly Total Claims
        axes[0].plot(months, monthly_stats["TotalClaims_Sum"], marker="o", linewidth=2, 
                    markersize=8, color="steelblue", label="Total Claims")
        axes[0].fill_between(months, monthly_stats["TotalClaims_Sum"], alpha=0.3, color="steelblue")
        axes[0].set_title("Monthly Total Claims Trend", fontsize=16, fontweight="bold", pad=15)
        axes[0].set_xlabel("Month", fontsize=12)
        axes[0].set_ylabel("Total Claims", fontsize=12)
        axes[0].tick_params(axis="x", rotation=45)
        axes[0].grid(True, alpha=0.3, linestyle="--")
        axes[0].legend()
        
        # Monthly Loss Ratio
        axes[1].plot(months, monthly_stats["LossRatio"], marker="o", linewidth=2, 
                    markersize=8, color="crimson", label="Loss Ratio")
        axes[1].fill_between(months, monthly_stats["LossRatio"], alpha=0.3, color="crimson")
        axes[1].set_title("Monthly Loss Ratio Trend", fontsize=16, fontweight="bold", pad=15)
        axes[1].set_xlabel("Month", fontsize=12)
        axes[1].set_ylabel("Loss Ratio", fontsize=12)
        axes[1].tick_params(axis="x", rotation=45)
        axes[1].grid(True, alpha=0.3, linestyle="--")
        axes[1].legend()
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / "temporal_trends.png", dpi=300, bbox_inches="tight")
            print(f"Saved: {self.output_dir / 'temporal_trends.png'}")
        plt.close()
    
    def plot_correlation_matrix(self, columns: Optional[List[str]] = None, save: bool = True) -> None:
        """
        Plot correlation matrix for numerical variables.
        
        Parameters:
        -----------
        columns : List[str], optional
            Columns to include. If None, uses key numerical columns.
        save : bool
            Whether to save the figure
        """
        if columns is None:
            key_cols = ["TotalPremium", "TotalClaims", "CustomValueEstimate", 
                       "SumInsured", "CalculatedPremiumPerTerm"]
            columns = [col for col in key_cols if col in self.data.columns]
        
        if not columns:
            print("No valid columns found for correlation matrix")
            return
        
        corr_matrix = self.data[columns].corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", 
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title("Correlation Matrix - Key Financial Variables", 
                    fontsize=16, fontweight="bold", pad=20)
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / "correlation_matrix.png", dpi=300, bbox_inches="tight")
            print(f"Saved: {self.output_dir / 'correlation_matrix.png'}")
        plt.close()
    
    def plot_vehicle_risk_analysis(self, vehicle_risk: pd.DataFrame, top_n: int = 10, save: bool = True) -> None:
        """
        Plot top vehicle makes/models by risk.
        
        Parameters:
        -----------
        vehicle_risk : pd.DataFrame
            Vehicle risk analysis dataframe
        top_n : int
            Number of top vehicles to display
        save : bool
            Whether to save the figure
        """
        top_vehicles = vehicle_risk.head(top_n)
        top_vehicles["Make_Model"] = top_vehicles["Make"] + " " + top_vehicles["Model"]
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        # Total Claims
        axes[0].barh(range(len(top_vehicles)), top_vehicles["TotalClaims_Sum"], 
                    color="steelblue", edgecolor="black")
        axes[0].set_yticks(range(len(top_vehicles)))
        axes[0].set_yticklabels(top_vehicles["Make_Model"], fontsize=10)
        axes[0].set_xlabel("Total Claims", fontsize=12, fontweight="bold")
        axes[0].set_title(f"Top {top_n} Vehicles by Total Claims", fontsize=14, fontweight="bold")
        axes[0].grid(axis="x", alpha=0.3)
        axes[0].invert_yaxis()
        
        # Loss Ratio
        axes[1].barh(range(len(top_vehicles)), top_vehicles["LossRatio"], 
                    color="crimson", edgecolor="black")
        axes[1].set_yticks(range(len(top_vehicles)))
        axes[1].set_yticklabels(top_vehicles["Make_Model"], fontsize=10)
        axes[1].set_xlabel("Loss Ratio", fontsize=12, fontweight="bold")
        axes[1].set_title(f"Top {top_n} Vehicles by Loss Ratio", fontsize=14, fontweight="bold")
        axes[1].grid(axis="x", alpha=0.3)
        axes[1].invert_yaxis()
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / "vehicle_risk_analysis.png", dpi=300, bbox_inches="tight")
            print(f"Saved: {self.output_dir / 'vehicle_risk_analysis.png'}")
        plt.close()
    
    def plot_geographic_comparison(self, geo_data: pd.DataFrame, save: bool = True) -> None:
        """
        Plot geographic comparison of insurance metrics.
        
        Parameters:
        -----------
        geo_data : pd.DataFrame
            Geographic analysis dataframe
        save : bool
            Whether to save the figure
        """
        province_summary = (
            geo_data.groupby("Province")
            .agg({"LossRatio": "mean", "TotalClaims_Sum": "sum", "TotalPremium_Sum": "sum"})
            .sort_values("LossRatio", ascending=False)
        )
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 12))
        
        # Loss Ratio by Province
        axes[0].bar(range(len(province_summary)), province_summary["LossRatio"], 
                   color="steelblue", edgecolor="black")
        axes[0].set_xticks(range(len(province_summary)))
        axes[0].set_xticklabels(province_summary.index, rotation=45, ha="right")
        axes[0].set_ylabel("Average Loss Ratio", fontsize=12, fontweight="bold")
        axes[0].set_title("Average Loss Ratio by Province", fontsize=14, fontweight="bold")
        axes[0].grid(axis="y", alpha=0.3)
        
        # Total Premium by Province
        axes[1].bar(range(len(province_summary)), province_summary["TotalPremium_Sum"], 
                   color="forestgreen", edgecolor="black")
        axes[1].set_xticks(range(len(province_summary)))
        axes[1].set_xticklabels(province_summary.index, rotation=45, ha="right")
        axes[1].set_ylabel("Total Premium", fontsize=12, fontweight="bold")
        axes[1].set_title("Total Premium by Province", fontsize=14, fontweight="bold")
        axes[1].grid(axis="y", alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / "geographic_comparison.png", dpi=300, bbox_inches="tight")
            print(f"Saved: {self.output_dir / 'geographic_comparison.png'}")
        plt.close()
    
    def generate_all_visualizations(self, eda_analyzer, save: bool = True) -> None:
        """
        Generate all standard visualizations.
        
        Parameters:
        -----------
        eda_analyzer : EDAAnalyzer
            EDAAnalyzer instance with computed results
        save : bool
            Whether to save figures
        """
        print("\nGenerating visualizations...")
        
        # Loss ratio by province
        if eda_analyzer.loss_ratio_results:
            self.plot_loss_ratio_by_province(
                eda_analyzer.loss_ratio_results["by_province"], save=save
            )
        
        # Claims distribution
        self.plot_claims_distribution(save=save)
        
        # Temporal trends
        if eda_analyzer.temporal_trends is not None:
            self.plot_temporal_trends(eda_analyzer.temporal_trends, save=save)
        
        # Correlation matrix
        self.plot_correlation_matrix(save=save)
        
        # Vehicle risk
        if eda_analyzer.vehicle_risk is not None:
            self.plot_vehicle_risk_analysis(eda_analyzer.vehicle_risk, save=save)
        
        # Geographic comparison
        geo_data = eda_analyzer.analyze_geographic_trends()
        self.plot_geographic_comparison(geo_data, save=save)
        
        print(f"\nAll visualizations saved to {self.output_dir}/")

