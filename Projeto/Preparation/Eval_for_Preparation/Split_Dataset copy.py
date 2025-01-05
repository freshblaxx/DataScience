from pandas import read_csv, DataFrame, Series, concat, to_numeric, to_datetime
import pandas as pd
from sklearn.model_selection import train_test_split


file_tag = "Financial_scaled_minmax_replace"
data: DataFrame = read_csv(
    "/Users/pascalludwig/Documents/Master/Semester 1/Term 2/Data Science/Project/DataScience/Projeto/Preparation/ScalingFinancial_truncate__scaled_minmax.csv", na_values="", parse_dates=True, dayfirst=True
)

df: DataFrame = data.copy(deep=True)

X = df.drop(columns=['CLASS'])  # Features
y = df['CLASS']  # Target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

train_data = pd.concat([X_train, y_train], axis=1)  # Combine features and target for training data
test_data = pd.concat([X_test, y_test], axis=1)  # Combine features and target for testing data

# Save the splits to new files without altering the original
train_data.to_csv(f"{file_tag}_training_data.csv", index=False)
test_data.to_csv(f"{file_tag}_testing_data.csv", index=False)

print("Training and testing datasets have been saved.")
print("Training class distribution:")
print(y_train.value_counts())
print("\nTesting class distribution:")
print(y_test.value_counts())
