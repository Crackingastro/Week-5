import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.preprocessing import FunctionTransformer
import joblib
from pathlib import Path

Data_DIR = Path(__file__).parent.parent / "Data"

# Define the feature engineering function
def create_features(df):
    """Function to create features from raw data"""
    df = df.copy()
    # extract datetime features
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    df['TransactionHour'] = df['TransactionStartTime'].dt.hour
    df['TransactionDay'] = df['TransactionStartTime'].dt.day
    df['TransactionMonth'] = df['TransactionStartTime'].dt.month
    df['TransactionYear'] = df['TransactionStartTime'].dt.year
    # drop ID and timestamp columns
    return df.drop(
        ['TransactionId','BatchId','AccountId','SubscriptionId',
         'CustomerId','TransactionStartTime'],
        axis=1
    )

# updated column lists without aggregates
NUMERIC = ['CountryCode','Amount','Value','PricingStrategy']
DATETIME = ['TransactionHour','TransactionDay','TransactionMonth','TransactionYear']
CATEGORICAL = [
    'CurrencyCode','ProviderId',
    'ProductId','ProductCategory','ChannelId'
]

# Create pipeline using FunctionTransformer instead of FeatureBuilder class
pipeline = Pipeline([
    ('features', FunctionTransformer(create_features)),
    ('preprocessor', ColumnTransformer([
        ('num', Pipeline([
            ('imputer', KNNImputer(n_neighbors=3)),
            ('scaler', StandardScaler())
        ]), NUMERIC),
        ('dt', Pipeline([
            ('imputer', SimpleImputer(strategy='median'))
        ]), DATETIME),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ]), CATEGORICAL),
    ], remainder='drop'))
])

# Load and process data
df = pd.read_csv(Data_DIR /'raw/data.csv', parse_dates=['TransactionStartTime'])
y = df['FraudResult']
X = df.drop(columns='FraudResult')

pipeline.fit(X)
joblib.dump(pipeline, 'pipeline.pkl')
print("pipeline saved to pipeline.pkl")

Xp = pipeline.transform(X)
dfp = pd.DataFrame(Xp, columns=NUMERIC + DATETIME + CATEGORICAL)
dfp['FraudResult'] = y.values
dfp[DATETIME] = dfp[DATETIME].astype('int16')
dfp.to_csv(Data_DIR /'processed/data2.csv', index=False)

print("processed CSV saved to ../Data/processed/data.csv")