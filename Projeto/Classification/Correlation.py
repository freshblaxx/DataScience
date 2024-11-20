import os
import pandas as pd
from matplotlib.pyplot import figure, savefig, show
from seaborn import heatmap

# Set the file name of the dataset
file_name = "class_financial distress.csv"  # Change this to your other CSV file when needed
file_tag = "financial"

# Load the dataset
data = pd.read_csv("class_financial distress.csv")

# Identify numeric columns
numeric_columns = data.select_dtypes(include=["number"]).columns.tolist()

# Compute the correlation matrix for numeric variables
corr_mtx = data[numeric_columns].corr().abs()

# Ensure the output directory exists
output_dir = "/Users/pascalludwig/Documents/Master/Semester 1/Term 2/Data Science/Project/DataScience/Projeto/Charts"
os.makedirs(output_dir, exist_ok=True)

# Create and save the heatmap of the correlation matrix
figure(figsize=(12, 10))
heatmap(
    corr_mtx,
    xticklabels=numeric_columns,
    yticklabels=numeric_columns,
    annot=False,
    cmap="Blues",
    vmin=0,
    vmax=1,
)
file_tag = os.path.splitext(os.path.basename(file_name))[0]  # Use the file name without extension as the tag
savefig(f"{output_dir}/{file_tag}_correlation_analysis.png")
show()
