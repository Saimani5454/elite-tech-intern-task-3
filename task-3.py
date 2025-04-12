import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.datasets import fetch_openml
# Step 1: Data Extraction
def load_data():
"""Loads the Titanic dataset from OpenML."""
print("Loading data...")
data = fetch_openml(name='titanic', version=1, as_frame=True)
df = data.frame
return df
# Step 2: Data Transformation
def preprocess_data(df):
"""Preprocesses the data by handling missing values, scaling numerical features, and encoding categorical features."""
print("Preprocessing data...")
# Separate features and target
X = df.drop(columns=["survived"])
y = df["survived"]
# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Define column groups
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns
# Preprocessing pipelines
numeric_transformer = Pipeline(steps=[
('imputer', SimpleImputer(strategy='mean')),
('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[
('imputer', SimpleImputer(strategy='most_frequent')),
('onehot', OneHotEncoder(handle_unknown='ignore'))])
# Combine transformers in a column transformer
preprocessor = ColumnTransformer(
transformers=[
('num', numeric_transformer, numeric_features),
('cat', categorical_transformer, categorical_features)])
# Create preprocessing and training pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
# Fit and transform the training data
X_train_processed = pipeline.fit_transform(X_train)
X_test_processed = pipeline.transform(X_test)
return X_train_processed, X_test_processed, y_train, y_test
# Step 3: Data Loading and Preprocessing
def load_and_preprocess_data():
"""Loads and preprocesses the data."""
print("Loading and preprocessing data...")
df = load_data()
X_train_processed, X_test_processed, y_train, y_test = preprocess_data(df)
return X_train_processed, X_test_processed, y_train, y_test
# Execute the pipeline
if __name__ == "__main__":
X_train_processed, X_test_processed, y_train, y_test = load_and_preprocess_data()
print("Data preprocessing and loading completed.")
