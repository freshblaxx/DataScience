import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.pyplot import subplots, savefig, show
import os
from pandas import read_csv, DataFrame
from numpy import ndarray

file_tag = "arrests"
filename = "class_ny_arrests.csv"

data: DataFrame = read_csv(filename, na_values=["", " "])

# Sample a subset of the data entries for better computing
sampled_data = data.sample(n=100, random_state=1)  # Adjust the number of samples as needed

# Filter numeric columns
numeric_columns = sampled_data.select_dtypes(include=["number"]).columns.to_list()

if numeric_columns:
    n: int = len(numeric_columns)

    fig: Figure
    axs: ndarray
    fig, axs = subplots(n, n, figsize=(n * 4, n * 4), squeeze=False)  # Adjust the figure size as needed
    for i in range(n):
        var1: str = numeric_columns[i]
        for j in range(n):
            if i != j:
                var2: str = numeric_columns[j]
                axs[i, j].scatter(sampled_data[var1], sampled_data[var2], alpha=0.5)
                axs[i, j].set_xlabel(var1)
                axs[i, j].set_ylabel(var2)
                axs[i, j].set_title(f"{var1} vs {var2}")
            else:
                axs[i, j].axis('off')  # Turn off the diagonal subplots

    # Remove any empty subplots
    for i in range(n):
        for j in range(n):
            if i == j:
                fig.delaxes(axs[i, j])

    plt.tight_layout()
    os.makedirs("Projeto/charts", exist_ok=True)
    savefig(f"Projeto/charts/{file_tag}_sparsity_study.png")
    plt.close(fig)  # Close the figure to free up memory
else:
    print("Sparsity class: there are no numeric variables.")