import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def load_dataset(dataset_path):
    return pd.read_csv(dataset_path)

def check_columns(dataset):
    print("\nDataset columns:")
    print(dataset.columns)
def preprocess_dataset(dataset):
    if 'species' in dataset.columns:
        X = dataset.drop('species', axis=1)
        y = dataset['species']
    else:
        # Assuming the column names are different
        X = dataset.drop('variety', axis=1)  # Use the correct column name
        y = dataset['variety']  # Use the correct column name
    return X, y


def split_dataset(X, y):
    return train_test_split(X, y, test_size=0.4, random_state=42)

def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def run_pipeline():
    dataset_path = input("Enter the path to your dataset: ")
    dataset = load_dataset(dataset_path)
    
    check_columns(dataset)
    
    X, y = preprocess_dataset(dataset)
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    model = train_model(X_train, y_train)
    
    dataset_with_predictions = dataset.copy()
    dataset_with_predictions['predicted_species'] = model.predict(X)
    print("\nDataset with Predictions:")
    print(dataset_with_predictions.head())

    

if __name__ == "__main__":
    run_pipeline()
    

