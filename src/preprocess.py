import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess(path):
    df = pd.read_csv(path)

    encoders = {}   # store encoders

    # Create target column if needed
    if 'num' in df.columns:
        df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
        df.drop('num', axis=1, inplace=True)

    # Drop unnecessary columns
    df.drop(['id', 'dataset'], axis=1, errors='ignore', inplace=True)

    # Drop columns with too many missing values
    df.drop(['ca', 'thal', 'slope'], axis=1, errors='ignore', inplace=True)

    # Fill numerical columns
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        df[col] = df[col].fillna(df[col].median())

    # Fill categorical columns
    for col in df.select_dtypes(include=['object', 'string']).columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Encode categorical columns (FIXED)
    for col in df.select_dtypes(include=['object', 'string']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le   # store encoder per column

    return df, encoders