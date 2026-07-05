import numpy as np
import pandas as pd
from sktlearn.impute import KNNImputer


def load_and_secure_input(file_path):
    """MODULE 1: INPUT - Securing Fidelity."""
    print("--- Executing Module 1: Input Fidelity ---")
    # Load dataset
    df = pd.read_csv(r"C:\Users\lavan\OneDrive\Desktop\nancy\Dataset for Data Analytics - Sheet1.csv")

    # 1. Calculate missingness proportion per feature
    missing_props = df.isnull().mean()

    for col in df.columns:
        prop = missing_props[col]
        if prop == 0:
            continue

        print(f"Feature '{col}' missingness: {prop:.2%}")

        # Rule-Based Decision Matrix Optimization
        if prop < 0.05:
            # Row Deletion for minimal missingness (< 5%)
            print(f" -> Strategy: Row Deletion (<5%)")
            df = df.dropna(subset=[col])

        elif 0.05 <= prop <= 0.20:
            # Handle Skewed Numeric vs Categorical
            if pd.api.types.is_numeric_dtype(df[col]):
                print(f" -> Strategy: Global Median Imputation (5%-20%)")
                df[col] = df[col].fillna(df[col].median())
            else:
                print(f" -> Strategy: Mode Imputation (Categorical 5%-20%)")
                df[col] = df[col].fillna(df[col].mode()[0])

        elif prop > 0.20:
            # Multi-Dimensional KNN Estimation for heavily missing values (>20%)
            print(
                f" -> Strategy: KNN Imputation (>20%) [Computational Complexity O(N^2)]"
            )
            if pd.api.types.is_numeric_dtype(df[col]):
                imputer = KNNImputer(n_neighbors=5)
                # Using an auxiliary numeric block to assist the imputation mapping
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
            else:
                print(
                    f" -> Strategy: Fallback Mode for non-numeric feature '{col}' (>20%)"
                )
                df[col] = df[col].fillna(df[col].mode()[0])

    # 2. Identify and Neutralize Outliers via IQR Boundaries
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Capping outliers to neutralize variance inflation without losing data volume
        outliers_lower = df[col] < lower_bound
        outliers_upper = df[col] > upper_bound

        if outliers_lower.any() or outliers_upper.any():
            total_outliers = outliers_lower.sum() + outliers_upper.sum()
            print(
                f"Feature '{col}': Neutralizing {total_outliers} IQR anomalies."
            )
            df[col] = np.clip(df[col], lower_bound, upper_bound)

    return df


def processing_engine(df):
    """MODULE 2: PROCESS - Vectorized Engine & High-Fidelity Feature Extraction."""
    print("\n--- Executing Module 2: Vectorized Processing & Engineering ---")

    # Ensure we are handling numerical vectors cleanly
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) >= 3:
        # Feature Engineering 1: Interaction Term (Product of the first two numeric features)
        col1, col2 = numeric_cols[0], numeric_cols[1]
        df["feature_interaction_prod"] = df[col1] * df[col2]
        print(f" Engineered Feature 1: Interaction Product ({col1} * {col2})")

        # Feature Engineering 2: Relative Ratio Value (Adding minor epsilon to prevent zero-division error)
        df["feature_relative_ratio"] = df[col1] / (df[col2] + 1e-5)
        print(f" Engineered Feature 2: Relative Ratio ({col1} / {col2})")

        # Feature Engineering 3: Cumulative Row-wise Z-Score Component (Statistical Binning/Scaling)
        df["feature_magnitude_score"] = df[numeric_cols].sum(axis=1)
        print(
            " Engineered Feature 3: Magnitude Aggregation (Row-wise Vector Sum)"
        )
    else:
        # Fallback if the dataset contains very few numeric elements
        df["feature_idx_power"] = np.arange(len(df)) ** 2
        df["feature_log_idx"] = np.log1p(np.arange(len(df)))
        df["feature_sin_transform"] = np.sin(np.arange(len(df)))
        print(
            " Engineered 3 mathematical fallback vector features due to small numeric footprint."
        )

    return df


def output_and_serve(df):
    """MODULE 3: OUTPUT - Contract Delivery Preparation."""
    print("\n--- Executing Module 3: Output Structural Serving ---")

    # Final sanity check verifying complete mathematical cleanliness
    assert (
        df.isnull().sum().sum() == 0
    ), "Assertion Error: Production pipeline failed! NaN values detected."

    print("Success: Dataset is cleanly mapped to a real-numbered space.")
    print(f"Final Data Frame Shape: {df.shape}")
    return df


# --- PIPELINE ARCHITECTURE RUNNER ---
if __name__ == "Dataset for Data Analytics - sheet1.csv":
    # TODO: Place your dataset file name here
    DATASET_PATH = "your_dataset.csv"

    try:
        # Run Input-Process-Output Framework Engine
        clean_input = load_and_secure_input(DATASET_PATH)
        processed_data = processing_engine(clean_input)
        production_ready_dataset = output_and_serve(processed_data)

        # Export step for training estimators
        production_ready_dataset.to_csv(
            "production_ready_clean_data.csv", index=False
        )
        print(
            "\n Pipeline executed perfectly. File saved as 'production_ready_clean_data.csv'"
        )

    except FileNotFoundError:
        print(
            f"\n[Error]: Could not locate '{DATASET_PATH}'. Please place your CSV file in the same folder as this script and update the variable name."
        )