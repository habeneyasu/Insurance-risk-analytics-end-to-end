"""
Data Preprocessing Module for Machine Learning Models.
Handles missing data, feature engineering, and encoding.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional, Any
import warnings

warnings.filterwarnings("ignore")


class DataPreprocessor:
    """Class for preprocessing data for machine learning models."""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize DataPreprocessor.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input dataframe
        """
        self.data = data.copy()
        self.processed_data: Optional[pd.DataFrame] = None
        self.scaler: Optional[StandardScaler] = None
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.onehot_encoders: Dict[str, OneHotEncoder] = {}
        self.feature_columns: List[str] = []
        self.categorical_columns: List[str] = []
        self.numerical_columns: List[str] = []
    
    def handle_missing_values(self, strategy: str = 'median') -> None:
        """
        Handle missing values in the dataset.
        
        Parameters:
        -----------
        strategy : str
            Strategy for imputation: 'median', 'mean', 'mode', or 'drop'
        """
        print(f"\nHandling missing values using strategy: {strategy}")
        
        # Identify columns with missing values
        missing_cols = self.data.columns[self.data.isnull().any()].tolist()
        missing_info = self.data[missing_cols].isnull().sum()
        
        print(f"Columns with missing values: {len(missing_cols)}")
        for col, count in missing_info.items():
            pct = (count / len(self.data)) * 100
            print(f"  {col}: {count:,} ({pct:.2f}%)")
        
        # Handle based on strategy
        if strategy == 'drop':
            # Drop columns with >50% missing
            high_missing = [col for col in missing_cols if (self.data[col].isnull().sum() / len(self.data)) > 0.5]
            if high_missing:
                print(f"Dropping columns with >50% missing: {high_missing}")
                self.data = self.data.drop(columns=high_missing)
        
        # Impute numerical columns
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if self.data[col].isnull().any():
                if strategy == 'median':
                    self.data[col].fillna(self.data[col].median(), inplace=True)
                elif strategy == 'mean':
                    self.data[col].fillna(self.data[col].mean(), inplace=True)
                elif strategy == 'mode':
                    self.data[col].fillna(self.data[col].mode()[0], inplace=True)
        
        # Impute categorical columns
        categorical_cols = self.data.select_dtypes(include=['object', 'bool']).columns
        for col in categorical_cols:
            if self.data[col].isnull().any():
                if strategy == 'mode':
                    self.data[col].fillna(self.data[col].mode()[0], inplace=True)
                else:
                    self.data[col].fillna('Unknown', inplace=True)
    
    def engineer_features(self) -> None:
        """
        Create new features that might be relevant to predictions.
        """
        print("\nEngineering new features...")
        
        # Vehicle age (if RegistrationYear exists)
        if 'RegistrationYear' in self.data.columns:
            current_year = 2015  # Based on data period
            self.data['VehicleAge'] = current_year - pd.to_numeric(
                self.data['RegistrationYear'], errors='coerce'
            )
            self.data['VehicleAge'] = self.data['VehicleAge'].fillna(
                self.data['VehicleAge'].median()
            )
            print("  ✓ Created VehicleAge feature")
        
        # Claim indicator
        self.data['HasClaim'] = (self.data['TotalClaims'] > 0).astype(int)
        print("  ✓ Created HasClaim feature")
        
        # Premium to Sum Insured ratio
        if 'SumInsured' in self.data.columns and 'TotalPremium' in self.data.columns:
            self.data['PremiumToSumInsuredRatio'] = (
                self.data['TotalPremium'] / (self.data['SumInsured'] + 1)
            )
            print("  ✓ Created PremiumToSumInsuredRatio feature")
        
        # Margin (Premium - Claims)
        if 'TotalPremium' in self.data.columns and 'TotalClaims' in self.data.columns:
            self.data['Margin'] = self.data['TotalPremium'] - self.data['TotalClaims']
            print("  ✓ Created Margin feature")
        
        # Loss Ratio
        if 'TotalPremium' in self.data.columns and 'TotalClaims' in self.data.columns:
            self.data['LossRatio'] = (
                self.data['TotalClaims'] / (self.data['TotalPremium'] + 1)
            )
            print("  ✓ Created LossRatio feature")
        
        # Vehicle value categories (if CustomValueEstimate exists)
        if 'CustomValueEstimate' in self.data.columns:
            self.data['VehicleValueCategory'] = pd.cut(
                self.data['CustomValueEstimate'].fillna(self.data['CustomValueEstimate'].median()),
                bins=[0, 50000, 100000, 200000, np.inf],
                labels=['Low', 'Medium', 'High', 'VeryHigh']
            )
            print("  ✓ Created VehicleValueCategory feature")
    
    def encode_categorical_features(self, method: str = 'onehot', max_categories: int = 10) -> None:
        """
        Encode categorical features to numeric format.
        
        Parameters:
        -----------
        method : str
            Encoding method: 'onehot' or 'label'
        max_categories : int
            Maximum number of categories for one-hot encoding
        """
        print(f"\nEncoding categorical features using {method} encoding...")
        
        # Identify categorical columns
        categorical_cols = self.data.select_dtypes(include=['object', 'bool', 'category']).columns.tolist()
        
        # Remove target variables and ID columns
        exclude_cols = ['PolicyID', 'UnderwrittenCoverID', 'TotalClaims', 'TotalPremium', 
                       'CalculatedPremiumPerTerm', 'HasClaim', 'Margin', 'LossRatio']
        categorical_cols = [col for col in categorical_cols if col not in exclude_cols]
        
        self.categorical_columns = categorical_cols
        print(f"Found {len(categorical_cols)} categorical columns to encode")
        
        if method == 'label':
            # Label encoding for ordinal or low-cardinality categoricals
            for col in categorical_cols:
                if self.data[col].nunique() <= max_categories:
                    le = LabelEncoder()
                    self.data[col + '_encoded'] = le.fit_transform(
                        self.data[col].astype(str).fillna('Unknown')
                    )
                    self.label_encoders[col] = le
                    self.data = self.data.drop(columns=[col])
                    print(f"  ✓ Label encoded: {col}")
        
        elif method == 'onehot':
            # One-hot encoding
            for col in categorical_cols:
                if self.data[col].nunique() <= max_categories:
                    # One-hot encode
                    dummies = pd.get_dummies(
                        self.data[col].astype(str).fillna('Unknown'),
                        prefix=col,
                        drop_first=True
                    )
                    self.data = pd.concat([self.data, dummies], axis=1)
                    self.data = self.data.drop(columns=[col])
                    print(f"  ✓ One-hot encoded: {col} ({self.data[col].nunique()} categories)")
                else:
                    # For high cardinality, use label encoding
                    le = LabelEncoder()
                    self.data[col + '_encoded'] = le.fit_transform(
                        self.data[col].astype(str).fillna('Unknown')
                    )
                    self.label_encoders[col] = le
                    self.data = self.data.drop(columns=[col])
                    print(f"  ✓ Label encoded (high cardinality): {col}")
    
    def select_features(self, target: str, exclude_cols: List[str] = None) -> List[str]:
        """
        Select features for modeling.
        
        Parameters:
        -----------
        target : str
            Target variable name
        exclude_cols : List[str], optional
            Additional columns to exclude
        
        Returns:
        --------
        List[str]
            List of feature column names
        """
        if exclude_cols is None:
            exclude_cols = []
        
        # Default exclusions
        default_exclude = [
            'PolicyID', 'UnderwrittenCoverID', 'TransactionMonth',
            'TotalClaims', 'TotalPremium', 'CalculatedPremiumPerTerm',
            'HasClaim', 'Margin', 'LossRatio'
        ]
        
        all_exclude = list(set(default_exclude + exclude_cols + [target]))
        
        # Get all numeric columns
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Filter out excluded columns
        feature_cols = [col for col in numeric_cols if col not in all_exclude]
        
        self.feature_columns = feature_cols
        self.numerical_columns = feature_cols
        
        print(f"\nSelected {len(feature_cols)} features for modeling")
        
        return feature_cols
    
    def prepare_regression_data(self, target: str, filter_condition: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Prepare data for regression modeling.
        
        Parameters:
        -----------
        target : str
            Target variable name
        filter_condition : pd.Series, optional
            Boolean series to filter data (e.g., claims > 0)
        
        Returns:
        --------
        Tuple[pd.DataFrame, pd.Series, List[str]]
            Features, target, and feature names
        """
        data = self.data.copy()
        
        # Apply filter if provided
        if filter_condition is not None:
            data = data[filter_condition].copy()
            print(f"\nFiltered data: {len(data):,} records")
        
        # Select features
        feature_cols = self.select_features(target)
        
        # Prepare X and y
        X = data[feature_cols].fillna(0)
        y = data[target]
        
        # Remove any remaining infinite values
        X = X.replace([np.inf, -np.inf], 0)
        
        print(f"Prepared data: X shape {X.shape}, y shape {y.shape}")
        
        self.processed_data = X
        return X, y, feature_cols
    
    def prepare_classification_data(self, target: str) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Prepare data for classification modeling.
        
        Parameters:
        -----------
        target : str
            Target variable name (binary)
        
        Returns:
        --------
        Tuple[pd.DataFrame, pd.Series, List[str]]
            Features, target, and feature names
        """
        return self.prepare_regression_data(target)
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """
        Split data into training and testing sets.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features
        y : pd.Series
            Target
        test_size : float
            Proportion of test set
        random_state : int
            Random seed
        
        Returns:
        --------
        Tuple
            X_train, X_test, y_train, y_test
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"\nData split:")
        print(f"  Training set: {len(X_train):,} samples")
        print(f"  Test set: {len(X_test):,} samples")
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Scale features using StandardScaler.
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        X_test : pd.DataFrame
            Test features
        
        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame]
            Scaled training and test features
        """
        self.scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        print("✓ Features scaled using StandardScaler")
        
        return X_train_scaled, X_test_scaled

