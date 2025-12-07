"""
Data loading utilities for insurance risk analytics.
Object-oriented implementation.
"""

import pandas as pd
import os
from pathlib import Path
from typing import Optional, Dict, Any


class DataLoader:
    """Class for loading and managing insurance data."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize DataLoader.
        
        Parameters:
        -----------
        data_dir : str
            Directory containing data files
        """
        self.data_dir = Path(data_dir)
        self.data: Optional[pd.DataFrame] = None
        self.summary: Optional[Dict[str, Any]] = None
    
    def load_data(self, file_path: Optional[str] = None, file_name: str = "MachineLearningRating_v3.txt") -> pd.DataFrame:
        """
        Load insurance data from file.
        
        Parameters:
        -----------
        file_path : str, optional
            Full path to the data file. If None, looks in data_dir.
        file_name : str
            Name of the data file (default: MachineLearningRating_v3.txt)
        
        Returns:
        --------
        pd.DataFrame
            Loaded insurance data
        """
        if file_path is None:
            file_path = self.data_dir / file_name
        
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        print(f"Loading data from: {file_path}")
        
        # Load pipe-delimited file
        self.data = pd.read_csv(
            file_path,
            sep="|",
            low_memory=False,
            encoding="utf-8"
        )
        
        print(f"Data loaded successfully. Shape: {self.data.shape}")
        print(f"Columns: {len(self.data.columns)}")
        
        return self.data
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get basic summary statistics of the dataset.
        
        Returns:
        --------
        dict
            Dictionary containing summary information
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        self.summary = {
            "shape": self.data.shape,
            "columns": list(self.data.columns),
            "dtypes": self.data.dtypes.to_dict(),
            "memory_usage_mb": self.data.memory_usage(deep=True).sum() / 1024**2,
            "missing_values": self.data.isnull().sum().to_dict(),
            "missing_percentage": (self.data.isnull().sum() / len(self.data) * 100).to_dict(),
            "numerical_columns": list(self.data.select_dtypes(include=["int64", "float64"]).columns),
            "categorical_columns": list(self.data.select_dtypes(include=["object", "bool"]).columns),
        }
        
        return self.summary
    
    def print_summary(self) -> None:
        """Print data summary to console."""
        if self.summary is None:
            self.get_data_summary()
        
        print("\n" + "="*60)
        print("DATA SUMMARY")
        print("="*60)
        print(f"Shape: {self.summary['shape']}")
        print(f"Memory Usage: {self.summary['memory_usage_mb']:.2f} MB")
        print(f"\nNumerical Columns: {len(self.summary['numerical_columns'])}")
        print(f"Categorical Columns: {len(self.summary['categorical_columns'])}")
        print("\nMissing Values:")
        missing = {k: v for k, v in self.summary['missing_percentage'].items() if v > 0}
        if missing:
            for col, pct in sorted(missing.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {col}: {pct:.2f}%")
        else:
            print("  No missing values found")
        print("="*60 + "\n")
    
    def get_data(self) -> pd.DataFrame:
        """
        Get the loaded data.
        
        Returns:
        --------
        pd.DataFrame
            The loaded dataframe
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        return self.data
