"""
Main script for running Machine Learning Models (Task 4).
Implements claim severity prediction, premium optimization, and claim probability prediction.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from data.load_data import DataLoader
from models.data_preprocessing import DataPreprocessor
from models.model_builder import ModelBuilder
from models.model_interpretability import ModelInterpreter


class MLModelingPipeline:
    """Class for orchestrating the complete ML modeling pipeline."""
    
    def __init__(self, data_dir: str = "data", data_file: str = "MachineLearningRating_v3.txt"):
        """
        Initialize MLModelingPipeline.
        
        Parameters:
        -----------
        data_dir : str
            Directory containing data files
        data_file : str
            Name of the data file
        """
        self.data_dir = data_dir
        self.data_file = data_file
        self.data_loader: DataLoader = None
        self.data = None
        self.preprocessor: DataPreprocessor = None
        self.results: dict = {}
    
    def load_data(self) -> None:
        """Load and prepare data."""
        print("=" * 80)
        print("MACHINE LEARNING MODELING - INSURANCE RISK ANALYTICS")
        print("AlphaCare Insurance Solutions (ACIS)")
        print("=" * 80)
        
        print("\n[Step 1] Loading data...")
        self.data_loader = DataLoader(data_dir=self.data_dir)
        self.data = self.data_loader.load_data(file_name=self.data_file)
        print(f"Data loaded: {self.data.shape[0]:,} records, {self.data.shape[1]} features")
    
    def preprocess_data(self) -> None:
        """Preprocess data for modeling."""
        print("\n[Step 2] Preprocessing data...")
        self.preprocessor = DataPreprocessor(self.data)
        
        # Handle missing values
        self.preprocessor.handle_missing_values(strategy='median')
        
        # Engineer features
        self.preprocessor.engineer_features()
        
        # Encode categorical features
        self.preprocessor.encode_categorical_features(method='onehot', max_categories=10)
        
        print("✓ Data preprocessing completed")
    
    def build_claim_severity_model(self) -> dict:
        """
        Build model to predict claim severity (TotalClaims for policies with claims > 0).
        
        Returns:
        --------
        dict
            Model results and metrics
        """
        print("\n" + "=" * 80)
        print("MODEL 1: CLAIM SEVERITY PREDICTION")
        print("=" * 80)
        print("Target: TotalClaims (for policies with claims > 0)")
        print("Evaluation: RMSE, R² Score")
        
        # Filter to policies with claims
        claims_data = self.data[self.data['TotalClaims'] > 0].copy()
        print(f"\nFiltered to {len(claims_data):,} policies with claims")
        
        # Prepare data
        preprocessor = DataPreprocessor(claims_data)
        preprocessor.handle_missing_values(strategy='median')
        preprocessor.engineer_features()
        preprocessor.encode_categorical_features(method='onehot', max_categories=10)
        
        X, y, feature_names = preprocessor.prepare_regression_data('TotalClaims')
        X_train, X_test, y_train, y_test = preprocessor.split_data(X, y, test_size=0.2)
        
        # Build and train models
        model_builder = ModelBuilder(model_type='regression')
        
        models_to_train = ['LinearRegression', 'RandomForest', 'XGBoost']
        
        # Build models
        model_builder.build_linear_regression()
        model_builder.build_random_forest(n_estimators=100, max_depth=10)
        model_builder.build_xgboost(n_estimators=100, max_depth=6, learning_rate=0.1)
        
        results = {}
        
        for model_name in models_to_train:
            print(f"\n--- {model_name} ---")
            model_builder.train_model(model_name, X_train, y_train)
            y_pred = model_builder.predict(model_name, X_test)
            metrics = model_builder.evaluate_regression(model_name, y_test, y_pred)
            
            # Get feature importance for tree-based models
            if model_name in ['RandomForest', 'XGBoost']:
                importance_df = model_builder.get_feature_importance(model_name, feature_names, top_n=10)
                metrics['feature_importance'] = importance_df.to_dict('records')
            
            results[model_name] = {
                'metrics': metrics,
                'predictions': y_pred.tolist(),
                'y_test': y_test.tolist()
            }
        
        # Find best model
        best_model = min(results.keys(), key=lambda x: results[x]['metrics']['RMSE'])
        print(f"\n✓ Best Model: {best_model} (RMSE: {results[best_model]['metrics']['RMSE']:.2f})")
        
        # SHAP analysis for best tree-based model
        if best_model in ['RandomForest', 'XGBoost']:
            print(f"\n[SHAP Analysis] Analyzing {best_model}...")
            interpreter = ModelInterpreter(
                model_builder.models[best_model],
                X_test.sample(n=min(100, len(X_test)), random_state=42),
                model_name=f"{best_model}_ClaimSeverity"
            )
            interpreter.create_explainer()
            shap_importance = interpreter.get_feature_importance(top_n=10)
            interpreter.plot_summary(output_dir="reports/figures")
            
            results[best_model]['shap_importance'] = shap_importance.to_dict('records')
            results[best_model]['shap_insights'] = interpreter.generate_business_insights(top_n=5)
        
        self.results['claim_severity'] = {
            'models': results,
            'best_model': best_model,
            'feature_names': feature_names
        }
        
        return self.results['claim_severity']
    
    def build_premium_optimization_model(self) -> dict:
        """
        Build model to predict optimal premium (CalculatedPremiumPerTerm).
        
        Returns:
        --------
        dict
            Model results and metrics
        """
        print("\n" + "=" * 80)
        print("MODEL 2: PREMIUM OPTIMIZATION")
        print("=" * 80)
        print("Target: CalculatedPremiumPerTerm")
        print("Evaluation: RMSE, R² Score")
        
        # Prepare data
        X, y, feature_names = self.preprocessor.prepare_regression_data('CalculatedPremiumPerTerm')
        X_train, X_test, y_train, y_test = self.preprocessor.split_data(X, y, test_size=0.2)
        
        # Build and train models
        model_builder = ModelBuilder(model_type='regression')
        
        models_to_train = ['LinearRegression', 'RandomForest', 'XGBoost']
        
        # Build models
        model_builder.build_linear_regression()
        model_builder.build_random_forest(n_estimators=100, max_depth=10)
        model_builder.build_xgboost(n_estimators=100, max_depth=6, learning_rate=0.1)
        
        results = {}
        
        for model_name in models_to_train:
            print(f"\n--- {model_name} ---")
            model_builder.train_model(model_name, X_train, y_train)
            y_pred = model_builder.predict(model_name, X_test)
            metrics = model_builder.evaluate_regression(model_name, y_test, y_pred)
            
            # Get feature importance for tree-based models
            if model_name in ['RandomForest', 'XGBoost']:
                importance_df = model_builder.get_feature_importance(model_name, feature_names, top_n=10)
                metrics['feature_importance'] = importance_df.to_dict('records')
            
            results[model_name] = {
                'metrics': metrics,
                'predictions': y_pred.tolist(),
                'y_test': y_test.tolist()
            }
        
        # Find best model
        best_model = min(results.keys(), key=lambda x: results[x]['metrics']['RMSE'])
        print(f"\n✓ Best Model: {best_model} (RMSE: {results[best_model]['metrics']['RMSE']:.2f})")
        
        # SHAP analysis for best tree-based model
        if best_model in ['RandomForest', 'XGBoost']:
            print(f"\n[SHAP Analysis] Analyzing {best_model}...")
            interpreter = ModelInterpreter(
                model_builder.models[best_model],
                X_test.sample(n=min(100, len(X_test)), random_state=42),
                model_name=f"{best_model}_PremiumOptimization"
            )
            interpreter.create_explainer()
            shap_importance = interpreter.get_feature_importance(top_n=10)
            interpreter.plot_summary(output_dir="reports/figures")
            
            results[best_model]['shap_importance'] = shap_importance.to_dict('records')
            results[best_model]['shap_insights'] = interpreter.generate_business_insights(top_n=5)
        
        self.results['premium_optimization'] = {
            'models': results,
            'best_model': best_model,
            'feature_names': feature_names
        }
        
        return self.results['premium_optimization']
    
    def build_claim_probability_model(self) -> dict:
        """
        Build model to predict probability of claim occurring (binary classification).
        
        Returns:
        --------
        dict
            Model results and metrics
        """
        print("\n" + "=" * 80)
        print("MODEL 3: CLAIM PROBABILITY PREDICTION")
        print("=" * 80)
        print("Target: HasClaim (binary classification)")
        print("Evaluation: Accuracy, Precision, Recall, F1 Score, AUC-ROC")
        
        # Create HasClaim target if not exists
        if 'HasClaim' not in self.data.columns:
            self.data['HasClaim'] = (self.data['TotalClaims'] > 0).astype(int)
        
        # Prepare data
        X, y, feature_names = self.preprocessor.prepare_classification_data('HasClaim')
        X_train, X_test, y_train, y_test = self.preprocessor.split_data(X, y, test_size=0.2)
        
        # Build and train models
        model_builder = ModelBuilder(model_type='classification')
        
        models_to_train = ['RandomForest', 'XGBoost']
        
        # Build models (skip LinearRegression for classification in this implementation)
        model_builder.build_random_forest(n_estimators=100, max_depth=10)
        model_builder.build_xgboost(n_estimators=100, max_depth=6, learning_rate=0.1)
        
        results = {}
        
        for model_name in models_to_train:
            print(f"\n--- {model_name} ---")
            model_builder.train_model(model_name, X_train, y_train)
            y_pred = model_builder.predict(model_name, X_test)
            
            # Get prediction probabilities
            y_pred_proba = model_builder.models[model_name].predict_proba(X_test)[:, 1]
            
            metrics = model_builder.evaluate_classification(model_name, y_test, y_pred, y_pred_proba)
            
            # Get feature importance
            importance_df = model_builder.get_feature_importance(model_name, feature_names, top_n=10)
            metrics['feature_importance'] = importance_df.to_dict('records')
            
            results[model_name] = {
                'metrics': metrics,
                'predictions': y_pred.tolist(),
                'predictions_proba': y_pred_proba.tolist(),
                'y_test': y_test.tolist()
            }
        
        # Find best model
        best_model = max(results.keys(), key=lambda x: results[x]['metrics'].get('AUC_ROC', results[x]['metrics']['F1_Score']))
        print(f"\n✓ Best Model: {best_model}")
        
        # SHAP analysis for best model
        print(f"\n[SHAP Analysis] Analyzing {best_model}...")
        interpreter = ModelInterpreter(
            model_builder.models[best_model],
            X_test.sample(n=min(100, len(X_test)), random_state=42),
            model_name=f"{best_model}_ClaimProbability"
        )
        interpreter.create_explainer()
        shap_importance = interpreter.get_feature_importance(top_n=10)
        interpreter.plot_summary(output_dir="reports/figures")
        
        results[best_model]['shap_importance'] = shap_importance.to_dict('records')
        results[best_model]['shap_insights'] = interpreter.generate_business_insights(top_n=5)
        
        self.results['claim_probability'] = {
            'models': results,
            'best_model': best_model,
            'feature_names': feature_names
        }
        
        return self.results['claim_probability']
    
    def generate_report(self) -> None:
        """Generate comprehensive model comparison report."""
        print("\n" + "=" * 80)
        print("MODEL COMPARISON REPORT")
        print("=" * 80)
        
        report_file = Path("reports") / "ml_models_report.txt"
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("MACHINE LEARNING MODELS - COMPREHENSIVE REPORT\n")
            f.write("AlphaCare Insurance Solutions (ACIS)\n")
            f.write("=" * 80 + "\n\n")
            
            # Claim Severity Model
            if 'claim_severity' in self.results:
                f.write("MODEL 1: CLAIM SEVERITY PREDICTION\n")
                f.write("-" * 80 + "\n")
                result = self.results['claim_severity']
                f.write(f"Best Model: {result['best_model']}\n\n")
                
                for model_name, model_result in result['models'].items():
                    f.write(f"{model_name}:\n")
                    metrics = model_result['metrics']
                    f.write(f"  RMSE: {metrics['RMSE']:.2f}\n")
                    f.write(f"  R² Score: {metrics['R2_Score']:.4f}\n")
                    f.write(f"  MAE: {metrics['MAE']:.2f}\n\n")
                
                # SHAP insights
                best_model = result['best_model']
                if 'shap_insights' in result['models'][best_model]:
                    f.write("SHAP Business Insights:\n")
                    for insight in result['models'][best_model]['shap_insights']:
                        f.write(f"  - {insight}\n")
                f.write("\n")
            
            # Premium Optimization Model
            if 'premium_optimization' in self.results:
                f.write("MODEL 2: PREMIUM OPTIMIZATION\n")
                f.write("-" * 80 + "\n")
                result = self.results['premium_optimization']
                f.write(f"Best Model: {result['best_model']}\n\n")
                
                for model_name, model_result in result['models'].items():
                    f.write(f"{model_name}:\n")
                    metrics = model_result['metrics']
                    f.write(f"  RMSE: {metrics['RMSE']:.2f}\n")
                    f.write(f"  R² Score: {metrics['R2_Score']:.4f}\n")
                    f.write(f"  MAE: {metrics['MAE']:.2f}\n\n")
                
                # SHAP insights
                best_model = result['best_model']
                if 'shap_insights' in result['models'][best_model]:
                    f.write("SHAP Business Insights:\n")
                    for insight in result['models'][best_model]['shap_insights']:
                        f.write(f"  - {insight}\n")
                f.write("\n")
            
            # Claim Probability Model
            if 'claim_probability' in self.results:
                f.write("MODEL 3: CLAIM PROBABILITY PREDICTION\n")
                f.write("-" * 80 + "\n")
                result = self.results['claim_probability']
                f.write(f"Best Model: {result['best_model']}\n\n")
                
                for model_name, model_result in result['models'].items():
                    f.write(f"{model_name}:\n")
                    metrics = model_result['metrics']
                    f.write(f"  Accuracy: {metrics['Accuracy']:.4f}\n")
                    f.write(f"  Precision: {metrics['Precision']:.4f}\n")
                    f.write(f"  Recall: {metrics['Recall']:.4f}\n")
                    f.write(f"  F1 Score: {metrics['F1_Score']:.4f}\n")
                    if 'AUC_ROC' in metrics:
                        f.write(f"  AUC-ROC: {metrics['AUC_ROC']:.4f}\n")
                    f.write("\n")
                
                # SHAP insights
                best_model = result['best_model']
                if 'shap_insights' in result['models'][best_model]:
                    f.write("SHAP Business Insights:\n")
                    for insight in result['models'][best_model]['shap_insights']:
                        f.write(f"  - {insight}\n")
                f.write("\n")
        
        print(f"✓ Comprehensive report saved to: {report_file}")
    
    def run(self) -> None:
        """Execute the complete ML modeling pipeline."""
        self.load_data()
        self.preprocess_data()
        
        # Build all three models
        self.build_claim_severity_model()
        self.build_premium_optimization_model()
        self.build_claim_probability_model()
        
        # Generate report
        self.generate_report()
        
        print("\n" + "=" * 80)
        print("Machine Learning Modeling completed successfully!")
        print("=" * 80)


def main():
    """Main function to run ML modeling pipeline."""
    pipeline = MLModelingPipeline()
    pipeline.run()


if __name__ == "__main__":
    main()

