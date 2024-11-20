from numpy import ndarray
from pandas import read_csv, DataFrame
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.pyplot import subplots, savefig, show
import os

file_tag = "finance"
filename = "class_financial distress.csv"

data: DataFrame = read_csv(filename, na_values="")

vars: list = data.columns.to_list()
if vars:
    n: int = len(vars)
    chunk_size = 5  # Number of variables per chunk
    num_chunks = (n + chunk_size - 1) // chunk_size  # Calculate the number of chunks

    for chunk_index in range(num_chunks):
        start_index = chunk_index * chunk_size
        end_index = min(start_index + chunk_size, n)
        chunk_vars = vars[start_index:end_index]
        chunk_n = len(chunk_vars)

        fig: Figure
        axs: ndarray
        fig, axs = subplots(chunk_n, chunk_n, figsize=(chunk_n * 5, chunk_n * 5), squeeze=False)
        for i in range(chunk_n):
            var1: str = chunk_vars[i]
            for j in range(chunk_n):
                if i != j:
                    var2: str = chunk_vars[j]
                    axs[i, j].scatter(data[var1], data[var2], alpha=0.5)
                    axs[i, j].set_xlabel(var1)
                    axs[i, j].set_ylabel(var2)
                    axs[i, j].set_title(f"{var1} vs {var2}")
                else:
                    axs[i, j].axis('off')  # Turn off the diagonal subplots

        # Remove any empty subplots
        for i in range(chunk_n):
            for j in range(chunk_n):
                if i == j:
                    fig.delaxes(axs[i, j])

        plt.tight_layout()
        os.makedirs("Projeto/charts", exist_ok=True)
        savefig(f"Projeto/charts/{file_tag}_sparsity_study_chunk_{chunk_index + 1}.png")
        show()
else:
    print("Sparsity class: there are no variables.")