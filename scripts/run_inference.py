"""
Inference script - makes predictions on new data.

Usage:
    python scripts/run_inference.py [--input INPUT_CSV] [--output OUTPUT_CSV]
"""
import sys
import argparse
from pathlib import Path
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from titanic_ml.models.predict import PredictPipeline
from titanic_ml.features.build_features import apply_feature_engineering
from titanic_ml.config.settings import RAW_TEST_PATH
from titanic_ml.utils.logger import logging
from titanic_ml.utils.helpers import generate_kaggle_submission


def main():
    """Main inference pipeline."""
    parser = argparse.ArgumentParser(description='Run inference on Titanic test data')
    parser.add_argument(
        '--input', 
        type=str, 
        default=str(RAW_TEST_PATH),
        help='Path to input CSV file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='submissions/submission.csv',
        help='Path to output submission CSV'
    )
    
    args = parser.parse_args()
    
    try:
        logging.info("=" * 70)
        logging.info("STARTING TITANIC ML INFERENCE PIPELINE")
        logging.info("=" * 70)
        
        # Load test data
        logging.info(f"\n[1/3] Loading test data from {args.input}...")
        test_df = pd.read_csv(args.input)
        passenger_ids = test_df['PassengerId'].copy()
        logging.info(f"[OK] Loaded {len(test_df)} test samples")
        
        # Apply feature engineering
        logging.info("\n[2/3] Applying feature engineering...")
        test_df_processed, _, _ = apply_feature_engineering(test_df)
        logging.info(f"[OK] Features engineered: {test_df_processed.shape}")
        
        # Make predictions
        logging.info("\n[3/3] Making predictions...")
        pipeline = PredictPipeline()
        predictions, probabilities = pipeline.predict(test_df_processed) # pyright: ignore[reportUnusedVariable]
        logging.info(f"[OK] Generated {len(predictions)} predictions")
        
        # Save submission
        output_path = Path(args.output)
        generate_kaggle_submission(
            predictions=predictions,
            passenger_ids=passenger_ids,
            file_name=output_path.name,
            output_dir=str(output_path.parent)
        )
        
        # Print statistics
        survival_rate = (predictions == 1).mean()
        logging.info("\n" + "=" * 70)
        logging.info(f"[OK] INFERENCE COMPLETE!")
        logging.info(f"[OK] Predicted survival rate: {survival_rate:.2%}")
        logging.info(f"[OK] Submission saved to: {args.output}")
        logging.info("=" * 70)
        
    except Exception as e:
        logging.error(f"Inference pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
