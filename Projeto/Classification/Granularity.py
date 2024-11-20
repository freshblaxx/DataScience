from pandas import DataFrame, read_csv
from sklearn.preprocessing import LabelEncoder
from matplotlib.figure import Figure
from matplotlib.pyplot import subplots, savefig, show
from typing import List
import os
import numpy as np

filename = "class_financial distress.csv"
file_tag = "financial"

data: DataFrame = read_csv(filename, na_values="")
print(data.columns)  # Print the column names to debug

# Function to encode nominal (categorical) variables
def encode_nominal_variables(df: DataFrame, nominal_vars: List[str]) -> DataFrame:
    label_encoders = {}
    for var in nominal_vars:
        encoder = LabelEncoder()
        df[var + "_encoded"] = encoder.fit_transform(df[var])
        label_encoders[var] = encoder  # Store the encoder for potential reuse
    return df, label_encoders

# Function to split date-like variables into granular levels (e.g., year, month)
def derive_date_variables(df: DataFrame, date_var: str) -> DataFrame:
    # Example: If Time is represented in months, calculate year and month
    df[date_var + "_year"] = df[date_var] // 12
    df[date_var + "_month"] = df[date_var] % 12
    return df

# Function to analyze granularity of variables by creating bar plots
def analyse_granularity(data: DataFrame, var: str, levels: List[str]) -> None:
    cols = len(levels)
    fig: Figure
    axs: np.ndarray
    fig, axs = subplots(1, cols, figsize=(cols * 4, 4), squeeze=False)
    fig.suptitle(f"Granularity Study for {var}")

    for i in range(cols):
        counts = data[levels[i]].value_counts()
        axs[0, i].bar(counts.index, counts.values)
        axs[0, i].set_title(levels[i])
        axs[0, i].set_xlabel(levels[i])
        axs[0, i].set_ylabel("Count")

    # Save the plot as an image and display it
    output_dir = "Projeto/charts"
    os.makedirs(output_dir, exist_ok=True)
    savefig(f"{output_dir}/{var}_granularity.png")
    show()

# Main function
if __name__ == "__main__":
    # Load the dataset
    data = read_csv(
        filename,
        header=0,
        parse_dates=False,  # Time is not directly formatted as a date
    )

    # Identify and encode nominal variables
    nominal_vars = ["Company", "CLASS"]  # Update this list based on your actual column names
    data_encoded, encoders = encode_nominal_variables(data, nominal_vars)

    # Process the time variable (if it contains date-related information)
    date_var = "Time"
    data_with_dates = derive_date_variables(data_encoded, date_var)

    # Analyze granularity for the Time variable
    analyse_granularity(
        data_with_dates,
        var=date_var,
        levels=[f"{date_var}_year", f"{date_var}_month"]
    )

    print("Processing complete. Encoded data and visualizations saved.")