from pandas import read_csv, DataFrame, Series, concat, to_numeric, to_datetime
import pandas as pd
from sklearn.model_selection import train_test_split


file_tag = "Financial"
<<<<<<< HEAD
data: DataFrame = read_csv("/Users/dominikfrank/Desktop/University/Master/Semester 1/PII/Data Science/Code for Project/DataScience/Projeto/Preparation/Data Balancing/class_financial_over.csv", na_values="", parse_dates=True, dayfirst=True
=======
data: DataFrame = read_csv("Projeto\Preparation\Outliers\Financial_drop_outliers_cleaned_arrests.csv", na_values="", parse_dates=True, dayfirst=True
>>>>>>> 49e94bb9eb1a5615fce9e7411d8fb6246b14a6a5
)

df: DataFrame = data.copy(deep=True)

X = df.drop(columns=['CLASS'])  # Features
y = df['CLASS']  # Target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

train_data = pd.concat([X_train, y_train], axis=1)  # Combine features and target for training data
test_data = pd.concat([X_test, y_test], axis=1)  # Combine features and target for testing data
# Save the splits to new files without altering the original
train_data.to_csv(f"{file_tag}_training_data.csv", index=False)
test_data.to_csv(f"{file_tag}_testing_data.csv", index=False)

print("Training and testing datasets have been saved.")