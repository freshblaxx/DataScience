from numpy import ndarray
from pandas import DataFrame, read_csv, to_datetime
from sklearn.preprocessing import LabelEncoder
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots, savefig, show
from typing import List
import os

filename = "class_financial distress.csv"
data = read_csv(filename, na_values="")

# Function to encode nominal (categorical) variables
def encode_nominal_variables(df: DataFrame, nominal_vars: List[str]) -> DataFrame:
    label_encoders = {}
    for var in nominal_vars:
        if var in df.columns:
            encoder = LabelEncoder()
            df[var + "_encoded"] = encoder.fit_transform(df[var])
            label_encoders[var] = encoder  # Store the encoder for potential reuse
    return df, label_encoders

# Function to split date-like variables into granular levels (e.g., year, month)
def derive_date_variables(df: DataFrame, date_var: str) -> DataFrame:
    if date_var in df.columns:
        df[date_var] = to_datetime(df[date_var], errors='coerce')
        df[date_var + "_year"] = df[date_var].dt.year
        df[date_var + "_month"] = df[date_var].dt.month
    return df

# Function to analyze granularity of variables by creating bar plots
def analyse_granularity(data: DataFrame, var: str, levels: List[str]) -> None:
    cols = len(levels)
    fig: Figure
    axs: ndarray
    fig, axs = subplots(1, cols, figsize=(cols * 4, 4), squeeze=False)
    fig.suptitle(f"Granularity Study for {var}")

    for i in range(cols):
        counts = data[levels[i]].value_counts()
        axs[0, i].bar(counts.index, counts.values)
        axs[0, i].set_title(levels[i])
        axs[0, i].set_xlabel(levels[i])
        axs[0, i].set_ylabel("Count")

    # Save the plot as an image and display it
    os.makedirs("projeto/charts", exist_ok=True)
    savefig(f"projeto/charts/{var}_granularity.png")
    show()

# Main function
if __name__ == "__main__":
    # Load the dataset
    data = read_csv(filename, na_values="")

    # Encode nominal variables
    nominal_vars = ["Company", "Time"]  # Replace with actual column names
    data, encoders = encode_nominal_variables(data, nominal_vars)

    # Derive date variables
    date_var = "Time"  # Replace with actual date column name
    data = derive_date_variables(data, date_var)

    # Analyze granularity
    levels = [date_var + "_year", date_var + "_month"]
    analyse_granularity(data, date_var, levels)

    print("Processing complete. Encoded data and visualizations saved.")