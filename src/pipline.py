import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer

def process_transaction_data(input_path: str, output_path: str, preview: bool = False) -> pd.DataFrame:
    """Process transaction data with robust missing value handling.
    
    Args:
        input_path: Path to raw data CSV
        output_path: Path to save processed data
        preview: Whether to show preview of processed data
        
    Returns:
        Processed DataFrame
    """
    
    def add_aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add customer-level aggregate features with proper missing value handling."""
        # First ensure Amount has no missing values for aggregation
        df['Amount'] = df['Amount'].fillna(0)
        
        agg_features = (
            df.groupby("CustomerId")["Amount"]
            .agg(
                Total_Transaction_Amount="sum",
                Average_Transaction_Amount="mean",
                Transaction_Count="count",
                Std_Transaction_Amount="std",
            )
            .reset_index()
        )
        
        # Handle cases where std is NaN (when only 1 transaction)
        agg_features['Std_Transaction_Amount'] = agg_features['Std_Transaction_Amount'].fillna(0)
        
        return df.merge(agg_features, on="CustomerId", how="left")

    def extract_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
        """Extract datetime features with missing value handling."""
        # Ensure datetime column has no missing values
        if df["TransactionStartTime"].isna().any():
            print("Warning: Found missing TransactionStartTime - filling with most recent date")
            df["TransactionStartTime"] = df["TransactionStartTime"].fillna(method='ffill')
        
        dt_col = pd.to_datetime(df["TransactionStartTime"])
        df["TransactionHour"] = dt_col.dt.hour.astype('int8')
        df["TransactionDay"] = dt_col.dt.day.astype('int8')
        df["TransactionMonth"] = dt_col.dt.month.astype('int8')  
        df["TransactionYear"] = dt_col.dt.year.astype('int16')   
        return df

    def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in specific columns before preprocessing."""
        # CurrencyCode - fill with mode
        if 'CurrencyCode' in df.columns:
            df['CurrencyCode'] = df['CurrencyCode'].fillna(df['CurrencyCode'].mode()[0])
        
        # CountryCode - fill with mode
        if 'CountryCode' in df.columns:
            df['CountryCode'] = df['CountryCode'].fillna(df['CountryCode'].mode()[0])
        
        # Numeric fields - fill with 0
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(0)
            
        return df

    print("Loading and preprocessing data...")
    try:
        df = pd.read_csv(input_path, parse_dates=["TransactionStartTime"])
    except Exception as e:
        raise ValueError(f"Error loading data: {str(e)}")
    
    # Initial missing value handling
    df = handle_missing_values(df)
    
    # Feature engineering
    df = add_aggregate_features(df)
    df = extract_datetime_features(df)

    # Separate target
    if 'FraudResult' not in df.columns:
        raise ValueError("Target column 'FraudResult' not found in data")
    y = df["FraudResult"]

    # Columns to drop
    drop_cols = [
        "TransactionId", "BatchId", "AccountId",
        "SubscriptionId", "CustomerId", "TransactionStartTime",
        "FraudResult"  # target
    ]
    X = df.drop(columns=[col for col in drop_cols if col in df.columns])

    # Identify feature types
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    all_numeric_features = X.select_dtypes(include=["number"]).columns.tolist()

    datetime_features = ["TransactionHour", "TransactionDay", "TransactionMonth", "TransactionYear"]
    regular_numeric_features = [f for f in all_numeric_features if f not in datetime_features]

    # --- Build robust preprocessing pipeline ---
    numeric_transformer = Pipeline([
        ("imputer", KNNImputer(n_neighbors=5, add_indicator=True)),  # Adds missing indicator
        ("scaler", StandardScaler())
    ])

    datetime_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True))
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent", add_indicator=True)),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, regular_numeric_features),
        ("dt", datetime_transformer, datetime_features),
        ("cat", categorical_transformer, categorical_features)
    ], remainder='passthrough')  # Keep any other columns

    print("Transforming data with missing value handling...")
    try:
        X_processed = preprocessor.fit_transform(X)
    except Exception as e:
        raise ValueError(f"Error during preprocessing: {str(e)}")

    # Get feature names with missing indicators
    num_features = [f"num_{f}" for f in regular_numeric_features]
    dt_features = [f"dt_{f}" for f in datetime_features]
    cat_features = [f"cat_{f}" for f in categorical_features]
    
    # Add missing indicator columns
    num_missing = [f"{f}_missing" for f in regular_numeric_features]
    dt_missing = [f"{f}_missing" for f in datetime_features]
    cat_missing = [f"{f}_missing" for f in categorical_features]
    
    all_features = num_features + num_missing + dt_features + dt_missing + cat_features + cat_missing
    
    df_processed = pd.DataFrame(X_processed, columns=all_features, index=df.index)
    df_processed["FraudResult"] = y.values

    # Convert datetime features back to integers
    for col in dt_features:
        df_processed[col] = df_processed[col].astype('int')

    # Show missing value summary
    print("\nMissing values after processing:")
    print(df_processed.isna().sum())

    # Show preview if requested
    if preview:
        print("\nProcessed Data Preview:")
        display(df_processed.head())

    # Save processed data
    df_processed.to_csv(output_path, index=False)
    print(f"\nSuccessfully saved processed data to {output_path}")
    return df_processed

# Example usage with error handling:
try:
    processed_data = process_transaction_data(
        "../Data/raw/data.csv", 
        "../Data/processed/data.csv", 
        preview=True
    )
except Exception as e:
    print(f"Error processing data: {str(e)}")

# Example usage:
process_transaction_data("../Data/raw/data.csv", "../Data/processed/data.csv", preview=True)