# run_all_models.py
"""
Master script to run the entire forecasting pipeline:
1. Data preprocessing
2. Feature engineering
3. Train all forecasting models
4. Create ensemble
5. Generate comparison report
"""

import sys
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")

def run_pipeline(skip_preprocessing=False, tune_hyperparameters=False):
    """
    Run the complete forecasting pipeline
    
    Parameters:
    - skip_preprocessing: Skip data preprocessing if data is already cleaned
    - tune_hyperparameters: Perform hyperparameter tuning (takes longer)
    """
    
    start_time = time.time()
    
    print_section("NEPAL RETAIL FORECASTING PIPELINE")
    print(f"Configuration:")
    print(f"  - Skip preprocessing: {skip_preprocessing}")
    print(f"  - Tune hyperparameters: {tune_hyperparameters}")
    print()
    
    # Step 1: Data Preprocessing
    if not skip_preprocessing:
        print_section("STEP 1/6: Data Preprocessing")
        try:
            from data_preprocessing import run as preprocess
            preprocess()
            print("✓ Data preprocessing completed successfully")
        except Exception as e:
            print(f"✗ Error in preprocessing: {e}")
            return False
    else:
        print_section("STEP 1/6: Data Preprocessing (SKIPPED)")
    
    # Step 2: Feature Engineering
    if not skip_preprocessing:
        print_section("STEP 2/6: Feature Engineering")
        try:
            from feature_engineering import run as engineer_features
            engineer_features()
            print("✓ Feature engineering completed successfully")
        except Exception as e:
            print(f"✗ Error in feature engineering: {e}")
            return False
    else:
        print_section("STEP 2/6: Feature Engineering (SKIPPED)")
    
    # Step 3: Train Prophet Model
    print_section("STEP 3/6: Training Prophet Model")
    try:
        from forecasting_model import run as train_prophet
        prophet_model, prophet_metrics, prophet_forecast = train_prophet(
            test_size=0.2,
            perform_cv=False  # Set to True for cross-validation (slower)
        )
        print("✓ Prophet model trained successfully")
    except Exception as e:
        print(f"✗ Error training Prophet: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 4: Train XGBoost Model
    print_section("STEP 4/6: Training XGBoost Model")
    try:
        from xgboost_model import run as train_xgboost
        xgb_model, xgb_metrics, xgb_forecast = train_xgboost(
            tune_hyperparameters=tune_hyperparameters,
            test_size=0.2
        )
        print("✓ XGBoost model trained successfully")
    except Exception as e:
        print(f"✗ Error training XGBoost: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 5: Train LightGBM Model
    print_section("STEP 5/6: Training LightGBM Model")
    try:
        from lightgbm_model import run as train_lightgbm
        lgbm_model, lgbm_metrics, lgbm_forecast = train_lightgbm(
            tune_hyperparameters=tune_hyperparameters,
            test_size=0.2
        )
        print("✓ LightGBM model trained successfully")
    except Exception as e:
        print(f"✗ Error training LightGBM: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 6: Create Ensemble
    print_section("STEP 6/6: Creating Ensemble Model")
    try:
        from esemble_model import run as create_ensemble
        ensemble_df, weights, metrics, comparison = create_ensemble(
            method='inverse_mae'  # Options: 'inverse_mae', 'inverse_rmse', 'equal'
        )
        print("✓ Ensemble model created successfully")
    except Exception as e:
        print(f"✗ Error creating ensemble: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Final Summary
    end_time = time.time()
    duration = end_time - start_time
    
    print_section("PIPELINE COMPLETED SUCCESSFULLY! 🎉")
    print(f"Total execution time: {duration/60:.2f} minutes")
    print()
    print("Model Performance Summary (Test Set):")
    print("-" * 70)
    print(f"{'Model':<15} {'MAE (NPR)':<20} {'RMSE (NPR)':<20} {'MAPE (%)':<15}")
    print("-" * 70)
    
    if prophet_metrics:
        print(f"{'Prophet':<15} {prophet_metrics['MAE']:>18,.2f} {prophet_metrics['RMSE']:>18,.2f} {prophet_metrics['MAPE']:>13.2f}")
    if xgb_metrics:
        print(f"{'XGBoost':<15} {xgb_metrics['MAE']:>18,.2f} {xgb_metrics['RMSE']:>18,.2f} {xgb_metrics['MAPE']:>13.2f}")
    if lgbm_metrics:
        print(f"{'LightGBM':<15} {lgbm_metrics['MAE']:>18,.2f} {lgbm_metrics['RMSE']:>18,.2f} {lgbm_metrics['MAPE']:>13.2f}")
    
    print("-" * 70)
    print()
    print("Generated Files:")
    print("  ✓ data/processed/prophet_forecast.csv")
    print("  ✓ data/processed/xgboost_forecast.csv")
    print("  ✓ data/processed/lightgbm_forecast.csv")
    print("  ✓ data/processed/ensemble_forecast.csv")
    print("  ✓ data/processed/model_comparison.csv")
    print("  ✓ models/prophet_metadata.json")
    print("  ✓ models/xgboost_metadata.json")
    print("  ✓ models/lightgbm_metadata.json")
    print("  ✓ models/ensemble_metadata.json")
    print()
    print("Next Steps:")
    print("  1. Review model_comparison.csv to see which model performed best")
    print("  2. Run the Streamlit dashboard: streamlit run app/main_improved.py")
    print("  3. Adjust hyperparameters if needed and re-run")
    print()
    
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run the complete forecasting pipeline')
    parser.add_argument(
        '--skip-preprocessing',
        action='store_true',
        help='Skip data preprocessing and feature engineering (use if data already processed)'
    )
    parser.add_argument(
        '--tune',
        action='store_true',
        help='Perform hyperparameter tuning (takes significantly longer)'
    )
    
    args = parser.parse_args()
    
    success = run_pipeline(
        skip_preprocessing=args.skip_preprocessing,
        tune_hyperparameters=args.tune
    )
    
    if not success:
        print("\n⚠️  Pipeline failed. Please check the error messages above.")
        sys.exit(1)
    else:
        print("✅ Pipeline completed successfully!")
        sys.exit(0)