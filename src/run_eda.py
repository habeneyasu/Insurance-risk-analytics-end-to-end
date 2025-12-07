"""
Main script for running Exploratory Data Analysis (EDA).
Uses object-oriented classes for modular and maintainable code.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from data.load_data import DataLoader
from analysis.data_quality import DataQualityChecker
from analysis.eda import EDAAnalyzer
from analysis.visualizations import VisualizationGenerator


def main():
    """Main function to run EDA pipeline."""
    
    print("="*70)
    print("INSURANCE RISK ANALYTICS - EXPLORATORY DATA ANALYSIS")
    print("AlphaCare Insurance Solutions (ACIS)")
    print("="*70)
    
    # Step 1: Load Data
    print("\n[Step 1] Loading data...")
    data_loader = DataLoader(data_dir="data")
    df = data_loader.load_data(file_name="MachineLearningRating_v3.txt")
    data_loader.get_data_summary()
    data_loader.print_summary()
    
    # Step 2: Data Quality Assessment
    print("\n[Step 2] Assessing data quality...")
    quality_checker = DataQualityChecker(df)
    quality_checker.generate_quality_report()
    quality_checker.print_quality_report()
    
    # Step 3: Exploratory Data Analysis
    print("\n[Step 3] Performing exploratory data analysis...")
    eda_analyzer = EDAAnalyzer(df)
    
    # Calculate loss ratios
    print("  - Calculating loss ratios...")
    eda_analyzer.calculate_loss_ratio()
    eda_analyzer.print_loss_ratio_summary()
    
    # Analyze distributions
    print("  - Analyzing distributions...")
    key_numerical_cols = ["TotalPremium", "TotalClaims", "CustomValueEstimate", 
                         "SumInsured", "CalculatedPremiumPerTerm"]
    distributions = eda_analyzer.analyze_distributions(
        numerical_cols=[col for col in key_numerical_cols if col in df.columns]
    )
    print(f"  - Analyzed distributions for {len(distributions)} numerical columns")
    
    # Analyze temporal trends
    print("  - Analyzing temporal trends...")
    temporal_trends = eda_analyzer.analyze_temporal_trends()
    print(f"  - Found {len(temporal_trends)} months of data")
    
    # Analyze vehicle risk
    print("  - Analyzing vehicle risk...")
    try:
        vehicle_risk = eda_analyzer.analyze_vehicle_risk()
        print(f"  - Analyzed {len(vehicle_risk)} unique vehicle make/model combinations")
        print(f"  - Top 5 highest risk vehicles:")
        for idx, row in vehicle_risk.head(5).iterrows():
            print(f"    {row['Make']} {row['Model']}: Loss Ratio = {row['LossRatio']:.4f}")
    except Exception as e:
        print(f"  - Warning: Could not analyze vehicle risk: {e}")
    
    # Analyze geographic trends
    print("  - Analyzing geographic trends...")
    geo_trends = eda_analyzer.analyze_geographic_trends()
    print(f"  - Analyzed {len(geo_trends)} province/postal code combinations")
    
    # Get descriptive statistics
    print("  - Computing descriptive statistics...")
    desc_stats = eda_analyzer.get_descriptive_statistics()
    print(f"  - Computed statistics for {len(desc_stats.columns)} numerical columns")
    
    # Step 4: Generate Visualizations
    print("\n[Step 4] Generating visualizations...")
    viz_generator = VisualizationGenerator(df, output_dir="reports/figures")
    viz_generator.generate_all_visualizations(eda_analyzer, save=True)
    
    # Step 5: Summary Report
    print("\n" + "="*70)
    print("EDA SUMMARY")
    print("="*70)
    print(f"Dataset Shape: {df.shape}")
    print(f"Overall Loss Ratio: {eda_analyzer.loss_ratio_results['overall_loss_ratio']:.4f}")
    print(f"Time Period: {temporal_trends['Month'].min()} to {temporal_trends['Month'].max()}")
    print(f"Number of Provinces: {df['Province'].nunique()}")
    print(f"Number of Unique Policies: {df['PolicyID'].nunique()}")
    print(f"Visualizations saved to: reports/figures/")
    print("="*70)
    
    print("\nEDA completed successfully!")


if __name__ == "__main__":
    main()

