"""
Feature engineering module - domain-specific feature creation.
"""

import sys
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd

from src.config.settings import TARGET_COLUMN
from src.utils.exception import CustomException
from src.utils.logger import logging

warnings.filterwarnings("ignore")


# Title mapping for consolidating rare titles
TITLE_MAPPING = {
    "Mr": "Mr",
    "Miss": "Miss",
    "Mrs": "Mrs",
    "Master": "Master",
    "Dr": "Rare",
    "Rev": "Rare",
    "Col": "Rare",
    "Major": "Rare",
    "Mlle": "Miss",
    "Mme": "Mrs",
    "Don": "Rare",
    "Dona": "Rare",
    "Lady": "Rare",
    "Countess": "Rare",
    "Jonkheer": "Rare",
    "Sir": "Rare",
    "Capt": "Rare",
    "Ms": "Miss",
    "the Countess": "Rare",
}

# Average fares by passenger class (calculated from training data)
FARE_BY_CLASS = {
    "1": 84.15,  # 1st class average
    "2": 20.66,  # 2nd class average
    "3": 13.68,  # 3rd class average
}


def infer_fare_from_class(pclass: str, family_size: int = 1) -> float:
    """
    Infer fare based on passenger class and family size.

    Args:
        pclass: Passenger class (1, 2, or 3)
        family_size: Size of family group

    Returns:
        Inferred fare value
    """
    base_fare = FARE_BY_CLASS.get(str(pclass), 32.2)

    # Apply family discount (groups typically got discounts)
    if family_size > 1:
        base_fare *= 0.9  # 10% discount for groups

    return base_fare


def apply_feature_engineering(  # noqa: C901
    df: pd.DataFrame, include_advanced: bool = True, is_training: bool = True
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Apply feature engineering transformations to the dataset.

    Basic Features:
    - cabin_multiple: Number of cabins purchased (wealth indicator)
    - name_title: Title extracted from name (social status)
    - norm_fare: Log-normalized fare (handles skewness)

    Advanced Features (if include_advanced=True):
    - family_size: Total family members aboard
    - is_alone: Whether passenger traveled alone
    - title_grouped: Consolidated title categories
    - age_group: Age categories
    - age_missing: Missing age indicator
    - fare_per_person: Fare divided by family size
    - fare_group: Fare quartiles
    - cabin_known: Whether cabin is known
    - cabin_deck: Deck letter from cabin
    - sex_pclass: Gender-class interaction
    - age_class: Age-class interaction

    Args:
        df: Input dataframe
        include_advanced: Whether to include advanced features
        is_training: Whether this is training data (affects column dropping)

    Returns:
        Tuple of (transformed_df, numerical_columns, categorical_columns)
    """
    try:
        logging.info(f"Starting feature engineering (advanced={include_advanced})")

        # Store original columns for reference
        has_name = "Name" in df.columns
        has_cabin = "Cabin" in df.columns
        has_ticket = "Ticket" in df.columns
        has_fare = "Fare" in df.columns

        # Basic Feature 1: Family size (needed for other features)
        if "SibSp" in df.columns and "Parch" in df.columns:
            df["family_size"] = df["SibSp"] + df["Parch"] + 1
            logging.info("Created family_size feature")

        # Basic Feature 2: Cabin multiple (wealth indicator)
        if has_cabin:
            df["cabin_multiple"] = df["Cabin"].apply(
                lambda x: 0 if pd.isna(x) else len(x.split(" "))
            )
            logging.info("Created cabin_multiple feature")
        else:
            df["cabin_multiple"] = 0

        # Basic Feature 3: Name title (social status)
        if has_name:
            df["name_title"] = df["Name"].apply(
                lambda x: x.split(",")[1].split(".")[0].strip() if "," in str(x) else "Unknown"
            )
            logging.info("Created name_title feature")

        # Basic Feature 4: Normalized fare (log transformation)
        if has_fare:
            # Infer missing fares from class and family size
            if df["Fare"].isna().any():
                for idx in df[df["Fare"].isna()].index:
                    pclass = str(df.loc[idx, "Pclass"])
                    family_size = (
                        int(df.loc[idx, "family_size"]) if "family_size" in df.columns else 1
                    )
                    df.loc[idx, "Fare"] = infer_fare_from_class(pclass, family_size)
                logging.info("Inferred missing fares from passenger class")

            df["norm_fare"] = np.log(df["Fare"] + 1)
            logging.info("Created norm_fare feature")

        # Advanced features
        if include_advanced:
            logging.info("Adding advanced features")

            # Feature: Is alone
            if "family_size" in df.columns:
                df["is_alone"] = (df["family_size"] == 1).astype(int)

            # Feature: Title grouping (consolidate rare titles)
            if "name_title" in df.columns:
                df["title_grouped"] = df["name_title"].map(TITLE_MAPPING).fillna("Rare")
                logging.info("Created title_grouped feature")

            # Feature: Age groups and age missing indicator
            if "Age" in df.columns:
                df["age_missing"] = df["Age"].isna().astype(int)
                df["age_group"] = pd.cut(
                    df["Age"].fillna(df["Age"].median()),
                    bins=[0, 12, 18, 35, 60, 100],
                    labels=["child", "teen", "adult", "middle_age", "senior"],
                )
                logging.info("Created age_group and age_missing features")

            # Feature: Fare per person
            if "norm_fare" in df.columns and "family_size" in df.columns:
                df["fare_per_person"] = df["norm_fare"] / df["family_size"]
                logging.info("Created fare_per_person feature")

            # Feature: Fare group (quartiles)
            if "Fare" in df.columns and len(df) > 1:
                try:
                    df["fare_group"] = pd.qcut(
                        df["Fare"],
                        q=4,
                        labels=["low", "medium", "high", "very_high"],
                        duplicates="drop",
                    )
                    logging.info("Created fare_group feature")
                except ValueError:
                    # If can't create quartiles (too few unique values), use simple binning
                    df["fare_group"] = pd.cut(
                        df["Fare"],
                        bins=[0, 7.9, 14.5, 31, float("inf")],
                        labels=["low", "medium", "high", "very_high"],
                    )
                    logging.info("Created fare_group feature (using fixed bins)")
            elif "Fare" in df.columns:
                # Single row - determine group based on fixed thresholds
                fare_val = df["Fare"].iloc[0]
                if fare_val < 7.9:
                    df["fare_group"] = "low"
                elif fare_val < 14.5:
                    df["fare_group"] = "medium"
                elif fare_val < 31:
                    df["fare_group"] = "high"
                else:
                    df["fare_group"] = "very_high"
                logging.info("Created fare_group feature (single row)")

            # Feature: Cabin known and cabin deck
            if has_cabin:
                df["cabin_known"] = df["Cabin"].notna().astype(int)
                df["cabin_deck"] = df["Cabin"].apply(
                    lambda x: str(x)[0] if pd.notna(x) else "Unknown"
                )
                logging.info("Created cabin_known and cabin_deck features")
            else:
                df["cabin_known"] = 0
                df["cabin_deck"] = "Unknown"

            # Feature: Ticket features
            if has_ticket:
                df["ticket_prefix"] = df["Ticket"].apply(
                    lambda x: x.split()[0] if " " in str(x) else "NONE"
                )
                df["ticket_has_letters"] = df["Ticket"].apply(
                    lambda x: int(any(c.isalpha() for c in str(x)))
                )
                logging.info("Created ticket features")

            # Feature: Interaction features
            if "Sex" in df.columns and "Pclass" in df.columns:
                df["sex_pclass"] = df["Sex"].astype(str) + "_" + df["Pclass"].astype(str)
                logging.info("Created sex_pclass interaction")

            if "Age" in df.columns and "Pclass" in df.columns:
                df["age_class"] = df["Age"].fillna(df["Age"].median()) * df["Pclass"].astype(int)
                logging.info("Created age_class interaction")

        # Handle missing Embarked values
        if "Embarked" in df.columns:
            df.dropna(subset=["Embarked"], inplace=True)
            logging.info("Dropped rows with missing Embarked values")

        # Drop original columns that have been engineered (only during training)
        if is_training:
            columns_to_drop = []
            if has_name:
                columns_to_drop.extend(["PassengerId", "Name"])
            if has_ticket:
                columns_to_drop.append("Ticket")
            if has_fare:
                columns_to_drop.append("Fare")
            if has_cabin:
                columns_to_drop.append("Cabin")

            df.drop(columns=columns_to_drop, inplace=True, errors="ignore")
            logging.info(f"Dropped original columns: {columns_to_drop}")

        # Identify numerical and categorical columns
        cols = df.columns.tolist()
        num_cols = list(df.select_dtypes("number").columns)
        cat_cols = list(set(cols) - set(num_cols))

        # Remove target from feature lists
        if TARGET_COLUMN in num_cols:
            num_cols.remove(TARGET_COLUMN)
        if TARGET_COLUMN in cat_cols:
            cat_cols.remove(TARGET_COLUMN)

        logging.info("Feature engineering complete:")
        logging.info(f"  Numerical features ({len(num_cols)}): {num_cols}")
        logging.info(f"  Categorical features ({len(cat_cols)}): {cat_cols}")

        return df, num_cols, cat_cols

    except Exception as e:
        logging.error(f"Error in feature engineering: {e}")
        raise CustomException(e, sys)


if __name__ == "__main__":
    import pandas as pd

    from src.config.settings import RAW_TRAIN_PATH
    from src.utils.logger import configure_logging, get_logger

    configure_logging(level="INFO")
    logger = get_logger(__name__)

    logger.info("Running feature engineering standalone")

    df = pd.read_csv(RAW_TRAIN_PATH)
    df_basic, num_cols, cat_cols = apply_feature_engineering(df.copy(), include_advanced=False)
    df_advanced, num_cols_adv, cat_cols_adv = apply_feature_engineering(
        df.copy(), include_advanced=True
    )

    logger.info(f"Original shape: {df.shape}")
    logger.info(f"Basic features shape: {df_basic.shape}")
    logger.info(f"Advanced features shape: {df_advanced.shape}")
    logger.info(f"Basic numerical features: {num_cols}")
    logger.info(f"Advanced numerical features: {num_cols_adv}")
