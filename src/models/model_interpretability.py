"""
Model Interpretability Module using SHAP.
Provides feature importance analysis and model explanations.
"""

import pandas as pd
import numpy as np
import shap
from typing import Dict, List, Tuple, Optional, Any
import warnings
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings("ignore")


class ModelInterpreter:
    """Class for model interpretability using SHAP."""
    
    def __init__(self, model: Any, X: pd.DataFrame, model_name: str = "Model"):
        """
        Initialize ModelInterpreter.
        
        Parameters:
        -----------
        model : Any
            Trained model
        X : pd.DataFrame
            Feature data (can be sample for tree models)
        model_name : str
            Name of the model
        """
        self.model = model
        self.X = X
        self.model_name = model_name
        self.explainer: Optional[Any] = None
        self.shap_values: Optional[np.ndarray] = None
        self.feature_names = X.columns.tolist()
    
    def create_explainer(self, sample_size: int = 100) -> None:
        """
        Create SHAP explainer for the model.
        
        Parameters:
        -----------
        sample_size : int
            Number of samples to use for explanation (for tree models)
        """
        print(f"\nCreating SHAP explainer for {self.model_name}...")
        
        # Sample data if too large
        if len(self.X) > sample_size:
            X_sample = self.X.sample(n=min(sample_size, len(self.X)), random_state=42)
        else:
            X_sample = self.X
        
        # Determine explainer type based on model
        model_type = type(self.model).__name__
        
        if 'XGB' in model_type or 'RandomForest' in model_type or 'GradientBoosting' in model_type:
            # Tree-based models
            self.explainer = shap.TreeExplainer(self.model)
            self.shap_values = self.explainer.shap_values(X_sample)
        elif 'Linear' in model_type:
            # Linear models
            self.explainer = shap.LinearExplainer(self.model, X_sample)
            self.shap_values = self.explainer.shap_values(X_sample)
        else:
            # Generic explainer
            self.explainer = shap.Explainer(self.model, X_sample)
            self.shap_values = self.explainer(X_sample).values
        
        print(f"  ✓ SHAP explainer created")
    
    def get_feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        """
        Get SHAP feature importance.
        
        Parameters:
        -----------
        top_n : int
            Number of top features to return
        
        Returns:
        --------
        pd.DataFrame
            Feature importance dataframe
        """
        if self.shap_values is None:
            self.create_explainer()
        
        # Handle multi-class case
        if len(self.shap_values.shape) > 2:
            shap_values_mean = np.abs(self.shap_values).mean(axis=0).mean(axis=0)
        else:
            shap_values_mean = np.abs(self.shap_values).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'shap_importance': shap_values_mean
        }).sort_values('shap_importance', ascending=False)
        
        print(f"\nTop {top_n} Features by SHAP Importance ({self.model_name}):")
        for idx, row in importance_df.head(top_n).iterrows():
            print(f"  {row['feature']}: {row['shap_importance']:.4f}")
        
        return importance_df
    
    def plot_summary(self, output_dir: str = "reports/figures", max_display: int = 10) -> None:
        """
        Create SHAP summary plot.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save plots
        max_display : int
            Maximum number of features to display
        """
        if self.shap_values is None:
            self.create_explainer()
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Sample data for plotting
        if len(self.X) > 100:
            X_plot = self.X.sample(n=100, random_state=42)
            if len(self.shap_values.shape) > 2:
                shap_values_plot = self.shap_values[:, :, 0] if self.shap_values.shape[2] == 1 else self.shap_values.mean(axis=2)
            else:
                shap_values_plot = self.shap_values
        else:
            X_plot = self.X
            shap_values_plot = self.shap_values
        
        # Create summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values_plot, X_plot, max_display=max_display, show=False)
        plt.title(f"SHAP Summary Plot - {self.model_name}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filename = f"shap_summary_{self.model_name.lower().replace(' ', '_')}.png"
        filepath = output_path / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ SHAP summary plot saved to: {filepath}")
    
    def plot_waterfall(self, instance_idx: int = 0, output_dir: str = "reports/figures") -> None:
        """
        Create SHAP waterfall plot for a single instance.
        
        Parameters:
        -----------
        instance_idx : int
            Index of instance to explain
        output_dir : str
            Directory to save plots
        """
        if self.shap_values is None:
            self.create_explainer()
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get SHAP values for this instance
        if len(self.shap_values.shape) > 2:
            shap_vals = self.shap_values[instance_idx, :, 0] if self.shap_values.shape[2] == 1 else self.shap_values[instance_idx, :, :].mean(axis=1)
        else:
            shap_vals = self.shap_values[instance_idx, :]
        
        # Create waterfall plot
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_vals,
                base_values=self.explainer.expected_value if hasattr(self.explainer, 'expected_value') else 0,
                data=self.X.iloc[instance_idx],
                feature_names=self.feature_names
            ),
            show=False
        )
        plt.title(f"SHAP Waterfall Plot - {self.model_name} (Instance {instance_idx})", 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filename = f"shap_waterfall_{self.model_name.lower().replace(' ', '_')}_instance_{instance_idx}.png"
        filepath = output_path / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ SHAP waterfall plot saved to: {filepath}")
    
    def get_feature_impact_analysis(self, top_n: int = 10) -> pd.DataFrame:
        """
        Analyze how features impact predictions.
        
        Parameters:
        -----------
        top_n : int
            Number of top features to analyze
        
        Returns:
        --------
        pd.DataFrame
            Feature impact analysis
        """
        if self.shap_values is None:
            self.create_explainer()
        
        # Calculate mean absolute SHAP values
        if len(self.shap_values.shape) > 2:
            mean_shap = np.abs(self.shap_values).mean(axis=0).mean(axis=0)
        else:
            mean_shap = np.abs(self.shap_values).mean(axis=0)
        
        # Calculate feature statistics
        feature_stats = []
        for i, feature in enumerate(self.feature_names):
            feature_data = self.X[feature]
            shap_vals = self.shap_values[:, i] if len(self.shap_values.shape) == 2 else self.shap_values[:, i, 0]
            
            # Calculate correlation between feature and SHAP values
            correlation = np.corrcoef(feature_data, shap_vals)[0, 1] if len(feature_data) > 1 else 0
            
            feature_stats.append({
                'feature': feature,
                'mean_abs_shap': mean_shap[i],
                'mean_feature_value': feature_data.mean(),
                'std_feature_value': feature_data.std(),
                'shap_feature_correlation': correlation
            })
        
        impact_df = pd.DataFrame(feature_stats).sort_values('mean_abs_shap', ascending=False)
        
        return impact_df.head(top_n)
    
    def generate_business_insights(self, top_n: int = 10) -> List[str]:
        """
        Generate business insights from SHAP analysis.
        
        Parameters:
        -----------
        top_n : int
            Number of top features to analyze
        
        Returns:
        --------
        List[str]
            List of business insight strings
        """
        impact_df = self.get_feature_impact_analysis(top_n)
        insights = []
        
        for idx, row in impact_df.iterrows():
            feature = row['feature']
            importance = row['mean_abs_shap']
            correlation = row['shap_feature_correlation']
            
            direction = "increases" if correlation > 0 else "decreases"
            insight = (
                f"SHAP analysis reveals that {feature} has an importance of {importance:.4f}. "
                f"The feature {direction} predicted values (correlation: {correlation:.3f}), "
                f"indicating its significant impact on model predictions."
            )
            insights.append(insight)
        
        return insights

