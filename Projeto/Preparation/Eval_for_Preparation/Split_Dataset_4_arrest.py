from pandas import read_csv, DataFrame, concat
import pandas as pd
from sklearn.model_selection import train_test_split

def split_and_save_data(file_path: str, file_tag: str, test_size: float = 0.30, random_state: int = 42):
    # Load data
    data: DataFrame = read_csv(file_path)
    
    # Split features and target
    X = data.drop(columns=['CLASS'])  # Features
    y = data['CLASS']  # Target

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Combine features and target for training and testing data
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    # Save to CSV
    train_data.to_csv(f"{file_tag}_training_data_arrest.csv", index=False)
    test_data.to_csv(f"{file_tag}_testing_data_arrest.csv", index=False)

    print(f"Datasets for {file_tag} saved: training and testing splits.")


# Paths to your datasets
file_path1 = "/Users/tomifemme/Desktop/DataScience/Projeto/Preparation/Scaling/arrests_scaled_minmax_arrest.csv"
file_tag1 = "MinMax_scaled_arrests"

file_path2 = "/Users/tomifemme/Desktop/DataScience/Projeto/Preparation/Scaling/arrests_scaled_zscore_arrest.csv"
file_tag2 = "Zscore_scaled_arrests"

# Split and save both datasets
split_and_save_data(file_path1, file_tag1)
split_and_save_data(file_path2, file_tag2)
