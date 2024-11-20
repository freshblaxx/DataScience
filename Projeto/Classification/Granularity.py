from numpy import ndarray
from pandas import DataFrame, read_csv, to_datetime
from sklearn.preprocessing import LabelEncoder
from matplotlib.figure import Figure
from matplotlib.pyplot import subplots, savefig, show
from typing import List
import os

# Load the dataset
filename = "class_ny_arrests.csv"  # Replace with the actual path
data = read_csv(filename, na_values="")
print(data.columns)
print(data.shape)

# Function to encode nominal (categorical) variables
def encode_nominal_variables(df: DataFrame, nominal_vars: List[str]) -> DataFrame:
    label_encoders = {}
    for var in nominal_vars:
        if var in df.columns:
            encoder = LabelEncoder()
            df[var + "_encoded"] = encoder.fit_transform(df[var].astype(str))
            label_encoders[var] = encoder  # Store the encoder for potential reuse
    return df, label_encoders

# Function to split date-like variables into granular levels (e.g., year, month)
def derive_date_variables(df: DataFrame, date_var: str) -> DataFrame:
    if date_var in df.columns:
        df[date_var] = to_datetime(df[date_var], format="%m/%d/%Y", errors='coerce')
        df[date_var + "_year"] = df[date_var].dt.year
        df[date_var + "_month"] = df[date_var].dt.month
        df[date_var + "_day"] = df[date_var].dt.day
    return df

# Function to analyze granularity of variables by creating bar plots
def analyse_granularity(data: DataFrame, var: str, levels: List[str]) -> None:
    cols = len(levels)
    fig: Figure
    axs: ndarray
    fig, axs = subplots(1, cols, figsize=(cols * 4, 4), squeeze=False)
    fig.suptitle(f"Granularity Study for {var}")

    for i in range(cols):
        if levels[i] in data.columns:
            counts = data[levels[i]].value_counts().sort_index()
            axs[0, i].bar(counts.index, counts.values)
            axs[0, i].set_title(levels[i])
            axs[0, i].set_xlabel(levels[i])
            axs[0, i].set_ylabel("Count")

    # Save the plot as an image and display it
    output_dir = "Projeto/Charts"
    os.makedirs(output_dir, exist_ok=True)
    savefig(f"{output_dir}/{var}_granularity.png")
    show()

# Main function
if __name__ == "__main__":
    # Define nominal variables for encoding
    nominal_vars = ["PD_DESC", "OFNS_DESC", "LAW_CODE", "LAW_CAT_CD", "ARREST_BORO", "AGE_GROUP", "PERP_SEX", "PERP_RACE"]

    # Encode nominal variables
    data, encoders = encode_nominal_variables(data, nominal_vars)

    # Derive date variables for granularity
    date_var = "ARREST_DATE"
    data = derive_date_variables(data, date_var)

    # Analyze granularity for ARREST_DATE
    levels = [f"{date_var}_year", f"{date_var}_month", f"{date_var}_day"]
    analyse_granularity(data, date_var, levels)

    # Additional analysis for encoded variables
    for var in nominal_vars:
        encoded_var = f"{var}_encoded"
        if encoded_var in data.columns:
            fig, ax = subplots(figsize=(6, 4))
            counts = data[encoded_var].value_counts().sort_index()
            ax.bar(counts.index, counts.values)
            ax.set_title(f"Distribution of {var}")
            ax.set_xlabel(f"{var} (Encoded)")
            ax.set_ylabel("Frequency")

            # Save and show the chart
            savefig(f"Projeto/Charts/{encoded_var}_distribution.png")
            show()