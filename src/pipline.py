import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer

def process_transaction_data(input_path: str, output_path: str, preview: bool = False) -> pd.DataFrame:

    # Helper function
    def add_aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add customer-level aggregate features."""
        agg_features = (
            df.groupby("CustomerId")["Amount"]
            .agg(
                Total_Transaction_Amount="sum",
                Average_Transaction_Amount="mean",
                Transaction_Count="count",
                Std_Transaction_Amount="std",
            )
            .fillna(0)
        )
        return df.merge(agg_features, on="CustomerId", how="left")

    def extract_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
        dt_col = df["TransactionStartTime"]
        df["TransactionHour"] = dt_col.dt.hour.astype('int8')
        df["TransactionDay"] = dt_col.dt.day.astype('int8')
        df["TransactionMonth"] = dt_col.dt.month.astype('int8')  
        df["TransactionYear"] = dt_col.dt.year.astype('int16')   
        return df

    print("Loading and preprocessing data...")
    df = pd.read_csv(input_path, parse_dates=["TransactionStartTime"])
    df = add_aggregate_features(df)
    df = extract_datetime_features(df)

    # Separate target
    y = df["FraudResult"]

    # Columns to drop
    drop_cols = [
        "TransactionId", "BatchId", "AccountId",
        "SubscriptionId", "CustomerId", "TransactionStartTime",
        "FraudResult"  # target
    ]
    X = df.drop(columns=drop_cols)

    # Identify feature types
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    all_numeric_features = X.select_dtypes(include=["number"]).columns.tolist()

    datetime_features = ["TransactionHour", "TransactionDay", "TransactionMonth", "TransactionYear"]
    regular_numeric_features = [f for f in all_numeric_features if f not in datetime_features]

    # --- Build preprocessing pipeline ---
    numeric_transformer = Pipeline([
        ("imputer", KNNImputer(n_neighbors=5)),
        ("scaler", StandardScaler())
    ])

    datetime_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))  # No scaling for datetime features
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, regular_numeric_features),
        ("dt", datetime_transformer, datetime_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    print("Transforming data...")
    X_processed = preprocessor.fit_transform(X)

    # Get feature names in correct order
    processed_cols = regular_numeric_features + datetime_features + categorical_features
    df_processed = pd.DataFrame(X_processed, columns=processed_cols, index=df.index)
    df_processed["FraudResult"] = y.values

    # Convert datetime features back to integers (in case imputation created floats)
    for col in datetime_features:
        df_processed[col] = df_processed[col].astype('int')

    # Show preview if requested
    if preview:
        print("\nProcessed Data Preview:")
        df_processed.head()

    # Save processed data
    df_processed.to_csv(output_path, index=False)
    print(f"\nSuccessfully saved processed data to {output_path}")
    return df_processed

# Example usage:
process_transaction_data("../Data/raw/data.csv", "../Data/processed/data.csv", preview=True)