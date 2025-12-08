"""
Model Building Module for Insurance Risk Analytics.
Implements Linear Regression, Random Forest, and XGBoost models.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
from typing import Dict, List, Tuple, Optional, Any
import warnings
import pickle
from pathlib import Path

warnings.filterwarnings("ignore")


class ModelBuilder:
    """Class for building and training machine learning models."""
    
    def __init__(self, model_type: str = 'regression'):
        """
        Initialize ModelBuilder.
        
        Parameters:
        -----------
        model_type : str
            Type of model: 'regression' or 'classification'
        """
        self.model_type = model_type
        self.models: Dict[str, Any] = {}
        self.predictions: Dict[str, np.ndarray] = {}
        self.metrics: Dict[str, Dict[str, float]] = {}
    
    def build_linear_regression(self, **kwargs) -> LinearRegression:
        """
        Build and return a Linear Regression model.
        
        Parameters:
        -----------
        **kwargs
            Additional parameters for LinearRegression
        
        Returns:
        --------
        LinearRegression
            Trained model
        """
        model = LinearRegression(**kwargs)
        self.models['LinearRegression'] = model
        return model
    
    def build_random_forest(self, n_estimators: int = 100, max_depth: int = 10, 
                          random_state: int = 42, **kwargs) -> Any:
        """
        Build and return a Random Forest model.
        
        Parameters:
        -----------
        n_estimators : int
            Number of trees
        max_depth : int
            Maximum depth of trees
        random_state : int
            Random seed
        **kwargs
            Additional parameters
        
        Returns:
        --------
        RandomForestRegressor or RandomForestClassifier
            Model instance
        """
        if self.model_type == 'regression':
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=-1,
                **kwargs
            )
        else:
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=-1,
                **kwargs
            )
        
        self.models['RandomForest'] = model
        return model
    
    def build_xgboost(self, n_estimators: int = 100, max_depth: int = 6,
                     learning_rate: float = 0.1, random_state: int = 42, **kwargs) -> Any:
        """
        Build and return an XGBoost model.
        
        Parameters:
        -----------
        n_estimators : int
            Number of boosting rounds
        max_depth : int
            Maximum depth of trees
        learning_rate : float
            Learning rate
        random_state : int
            Random seed
        **kwargs
            Additional parameters
        
        Returns:
        --------
        xgb.XGBRegressor or xgb.XGBClassifier
            Model instance
        """
        if self.model_type == 'regression':
            model = xgb.XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=random_state,
                n_jobs=-1,
                **kwargs
            )
        else:
            model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=random_state,
                n_jobs=-1,
                eval_metric='logloss',
                **kwargs
            )
        
        self.models['XGBoost'] = model
        return model
    
    def train_model(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Train a model.
        
        Parameters:
        -----------
        model_name : str
            Name of the model to train
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Build it first.")
        
        print(f"\nTraining {model_name}...")
        self.models[model_name].fit(X_train, y_train)
        print(f"  ✓ {model_name} trained successfully")
    
    def predict(self, model_name: str, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using a trained model.
        
        Parameters:
        -----------
        model_name : str
            Name of the model
        X : pd.DataFrame
            Features
        
        Returns:
        --------
        np.ndarray
            Predictions
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found.")
        
        predictions = self.models[model_name].predict(X)
        self.predictions[model_name] = predictions
        return predictions
    
    def evaluate_regression(self, model_name: str, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Evaluate regression model performance.
        
        Parameters:
        -----------
        model_name : str
            Name of the model
        y_true : pd.Series
            True target values
        y_pred : np.ndarray
            Predicted values
        
        Returns:
        --------
        dict
            Dictionary of metrics
        """
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mae = np.mean(np.abs(y_true - y_pred))
        
        metrics = {
            'RMSE': rmse,
            'R2_Score': r2,
            'MAE': mae
        }
        
        self.metrics[model_name] = metrics
        
        print(f"\n{model_name} Performance:")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  R² Score: {r2:.4f}")
        print(f"  MAE: {mae:.2f}")
        
        return metrics
    
    def evaluate_classification(self, model_name: str, y_true: pd.Series, y_pred: np.ndarray,
                                y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Evaluate classification model performance.
        
        Parameters:
        -----------
        model_name : str
            Name of the model
        y_true : pd.Series
            True target values
        y_pred : np.ndarray
            Predicted values
        y_pred_proba : np.ndarray, optional
            Predicted probabilities
        
        Returns:
        --------
        dict
            Dictionary of metrics
        """
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        metrics = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1_Score': f1
        }
        
        if y_pred_proba is not None and len(np.unique(y_true)) == 2:
            try:
                auc = roc_auc_score(y_true, y_pred_proba)
                metrics['AUC_ROC'] = auc
            except:
                pass
        
        self.metrics[model_name] = metrics
        
        print(f"\n{model_name} Performance:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        if 'AUC_ROC' in metrics:
            print(f"  AUC-ROC: {metrics['AUC_ROC']:.4f}")
        
        return metrics
    
    def get_feature_importance(self, model_name: str, feature_names: List[str], top_n: int = 10) -> pd.DataFrame:
        """
        Get feature importance from tree-based models.
        
        Parameters:
        -----------
        model_name : str
            Name of the model
        feature_names : List[str]
            List of feature names
        top_n : int
            Number of top features to return
        
        Returns:
        --------
        pd.DataFrame
            Feature importance dataframe
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found.")
        
        model = self.models[model_name]
        
        # Check if model has feature_importances_
        if not hasattr(model, 'feature_importances_'):
            raise ValueError(f"Model {model_name} does not support feature importance.")
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop {top_n} Features for {model_name}:")
        for idx, row in importance_df.head(top_n).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        return importance_df
    
    def save_model(self, model_name: str, filepath: str) -> None:
        """
        Save a trained model to disk.
        
        Parameters:
        -----------
        model_name : str
            Name of the model
        filepath : str
            Path to save the model
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found.")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.models[model_name], f)
        
        print(f"✓ Model saved to: {filepath}")
    
    def load_model(self, model_name: str, filepath: str) -> None:
        """
        Load a trained model from disk.
        
        Parameters:
        -----------
        model_name : str
            Name to assign to the loaded model
        filepath : str
            Path to the model file
        """
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        self.models[model_name] = model
        print(f"✓ Model loaded from: {filepath}")

