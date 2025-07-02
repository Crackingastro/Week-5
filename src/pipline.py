import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
import joblib

class FeatureBuilder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        df = X.copy()
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

pipeline = Pipeline([
    ('features', FeatureBuilder()),
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


df = pd.read_csv('../Data/raw/data.csv', parse_dates=['TransactionStartTime'])
y = df['FraudResult']
X = df.drop(columns='FraudResult')

pipeline.fit(X)
joblib.dump(pipeline, 'pipeline.pkl')
print("pipeline saved to pipeline.pkl")

Xp = pipeline.transform(X)
dfp = pd.DataFrame(Xp, columns=NUMERIC + DATETIME + CATEGORICAL)
dfp['FraudResult'] = y.values
dfp[DATETIME] = dfp[DATETIME].astype('int16')
dfp.to_csv('../Data/processed/data.csv', index=False)

print("processed CSV saved to ../Data/processed/data.csv")
