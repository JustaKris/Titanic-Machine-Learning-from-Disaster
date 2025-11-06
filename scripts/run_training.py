"""
Training script - trains and saves the best model.

Usage:
    python scripts/run_training.py
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import DataLoader
from src.data.transformer import DataTransformer
from src.models.train import ModelTrainer
from src.utils.logger import logging
from src.config.settings import setup_directories


def main():
    """Main training pipeline."""
    try:
        logging.info("=" * 70)
        logging.info("STARTING TITANIC ML TRAINING PIPELINE")
        logging.info("=" * 70)
        
        # Ensure directories exist
        setup_directories()
        
        # Step 1: Load data
        logging.info("\n[1/3] Loading and splitting data...")
        loader = DataLoader()
        train_path, test_path = loader.load_data()
        logging.info(f"[OK] Data loaded: train={train_path}, test={test_path}")
        
        # Step 2: Transform data
        logging.info("\n[2/3] Applying feature engineering and preprocessing...")
        transformer = DataTransformer()
        X_train, y_train, X_test, y_test, preprocessor_path = transformer.transform_data(
            train_path, test_path
        )
        logging.info(f"[OK] Data transformed: X_train shape={X_train.shape}")
        logging.info(f"[OK] Preprocessor saved: {preprocessor_path}")
        
        # Step 3: Train models
        logging.info("\n[3/3] Training and optimizing models...")
        trainer = ModelTrainer()
        best_model, best_score = trainer.train( # pyright: ignore[reportUnusedVariable]
            X_train, y_train, X_test, y_test, use_voting=True
        )
        
        logging.info("\n" + "=" * 70)
        logging.info(f"[OK] TRAINING COMPLETE!")
        logging.info(f"[OK] Best Model Score (F1): {best_score:.4f} ({best_score*100:.2f}%)")
        logging.info(f"[OK] Model saved to: {trainer.model_path}")
        logging.info("=" * 70)
        
        return best_score
        
    except Exception as e:
        logging.error(f"Training pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
