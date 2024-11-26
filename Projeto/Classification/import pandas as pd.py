import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Load the dataset
filename = "class_ny_arrests.csv"
data = pd.read_csv(filename, na_values="")

# Define the list of symbolic columns in your dataset
symbolic_columns = ['LAW_CAT_CD', 'ARREST_BORO', 'AGE_GROUP', 'PERP_SEX', 'PERP_RACE']

# Check if the symbolic columns exist in the dataset
valid_symbolic_columns = [col for col in symbolic_columns if col in data.columns]

# Create a PDF to save all plots
output_file = "symbolic_columns_histograms.pdf"
with PdfPages(output_file) as pdf:
    if len(valid_symbolic_columns) > 0:
        for column in valid_symbolic_columns:
            if data[column].notna().sum() > 0:  # Ensure the column has non-NaN values
                # Count occurrences for each category in the symbolic column
                class_distribution = data[column].value_counts()

                # Plot the histogram (bar chart) for the symbolic variable
                plt.figure(figsize=(8, 6))
                class_distribution.plot(kind='bar', color='skyblue', edgecolor='black')

                # Add labels and title
                plt.title(f"Class Distribution of Symbolic Variable '{column}'", fontsize=14)
                plt.xlabel("Class", fontsize=12)
                plt.ylabel("Number of Records", fontsize=12)

                # Annotate bar heights
                for index, value in enumerate(class_distribution):
                    plt.text(index, value + 0.5, str(value), ha='center', fontsize=10)

                # Save the current figure to the PDF
                pdf.savefig()
                plt.close()  # Close the plot to avoid displaying it in the interactive session
            else:
                print(f"Column '{column}' has no valid data to plot.")
    else:
        print("No valid symbolic columns found in the dataset.")

print(f"All plots have been saved to {output_file}")
