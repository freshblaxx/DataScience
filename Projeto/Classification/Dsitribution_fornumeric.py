from turtle import pd
import numpy as np
from numpy import ndarray
from pandas import DataFrame, read_csv, to_datetime
from sklearn.preprocessing import LabelEncoder
from matplotlib.figure import Figure
from matplotlib.pyplot import subplots, savefig, show
from scipy.stats import norm, expon, lognorm
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

# Function to compute known statistical distributions
def compute_known_distributions(x_values: list) -> dict:
    distributions = dict()
    
    # Ensure numeric values
    numeric_values = pd.to_numeric(x_values, errors='coerce').dropna()

    if len(numeric_values) < 2:  # Ensure there are enough points to fit distributions
        return distributions

    # Gaussian (Normal) Distribution
    mean, sigma = norm.fit(numeric_values)
    distributions[f"Normal({mean:.1f}, {sigma:.2f})"] = norm.pdf(
        np.linspace(min(numeric_values), max(numeric_values), 1000), mean, sigma
    )

    # Exponential Distribution
    loc, scale = expon.fit(numeric_values)
    distributions[f"Exp({1 / scale:.2f})"] = expon.pdf(
        np.linspace(min(numeric_values), max(numeric_values), 1000), loc, scale
    )

    # Log-Normal Distribution
    sigma, loc, scale = lognorm.fit(numeric_values)
    distributions[f"LogNor({np.log(scale):.1f}, {sigma:.2f})"] = lognorm.pdf(
        np.linspace(min(numeric_values), max(numeric_values), 1000), sigma, loc, scale
    )

    return distributions

# Function to plot histograms with distribution fits for numeric variables
def histogram_with_distributions(data: DataFrame, numeric_vars: List[str]) -> None:
    for var in numeric_vars:
        if var in data.columns:
            values = data[var].dropna().to_list()
            fig, ax = subplots(figsize=(8, 5))

            # Plot histogram
            ax.hist(values, density=True, color='skyblue', edgecolor='black', bins=30)
            
            # Compute and plot distributions
            distributions = compute_known_distributions(values)
            for label, distribution in distributions.items():
                ax.plot(np.linspace(min(values), max(values), 1000), distribution, label=label, linewidth=2)

            ax.set_title(f"Histogram and Distributions for {var}")
            ax.set_xlabel(var)
            ax.set_ylabel("Density")
            ax.legend()

            # Save and show the chart
            output_dir = "Projeto/Charts"
            os.makedirs(output_dir, exist_ok=True)
            savefig(f"{output_dir}/{var}_histogram.png")
            show()

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

    # Analyze numeric variables
    numeric_vars = data.select_dtypes(include=["number"]).columns.tolist()
    if numeric_vars:
        histogram_with_distributions(data, numeric_vars)
    else:
        print("No numeric variables found.")
