from pandas import read_csv, DataFrame, Series
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig, show, subplots
from scipy.stats import norm, expon, lognorm
import numpy as np
import pandas as pd



filename = "class_ny_arrests.csv"
file_tag = "arrests"

data: DataFrame = read_csv(filename, na_values="")
print(data.shape)


# NUMBER OF RECORDS AND VARIABLES

def plot_bar_chart(labels, values, title=""):
    plt.bar(labels, values, color='skyblue')
    plt.title(title)
    plt.xlabel("Categories")
    plt.ylabel("Values")

    
    for i, value in enumerate(values):
        plt.text(i, value + 0.001 * max(values), f'{value}', ha='center', va='bottom', fontsize=10)

    plt.show()


values = {"nr records": data.shape[0], "nr variables": data.shape[1]}
plot_bar_chart(list(values.keys()), list(values.values()), title="Nr of records vs nr variables")


# MISSING VALUES
missing_values = data.isna().sum()

def plot_bar_chart2(labels, values, title=""):
    plt.bar(labels, values, color='skyblue')
    plt.title(title)
    plt.xlabel("Variables")
    plt.ylabel("Number of Missing Values")
    
    
    for i, value in enumerate(values):
        plt.text(i, value + 0.001 * max(values), f'{value}', ha='center', va='bottom', fontsize=10)
    
   
    plt.xticks(rotation=90) 
    plt.tight_layout() 

    plt.show()

plot_bar_chart2(missing_values.index, missing_values.values, title="Number of Missing Values per Variable")


# Number of variables per type

variable_types = data.dtypes.value_counts()

def categorize_data_type(dtype):
    if pd.api.types.is_numeric_dtype(dtype):
        return "numeric"
    elif pd.api.types.is_bool_dtype(dtype):
        return "binary"
    elif pd.api.types.is_datetime64_any_dtype(dtype): 
        return "date"
    elif pd.api.types.is_object_dtype(dtype): 
        return "symbolic"
    else:
        return "other"

category_column_mapping = data.dtypes.apply(categorize_data_type)
variable_types = category_column_mapping.value_counts()

def plot_bar_chart3(labels, values, title=""):
    
    labels = [str(label) for label in labels]  
    plt.bar(labels, values, color='skyblue')
    plt.title(title)
    plt.xlabel("Data Types")
    plt.ylabel("Number of Variables")
    
    
    for i, value in enumerate(values):
        plt.text(i, value + 0.001 * max(values), f'{value}', ha='center', va='bottom', fontsize=10)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

plot_bar_chart3(variable_types.index, variable_types.values, title="Number of Variables per Data Type")


# Global boxplots
numeric_columns = data.select_dtypes(include=["number"]).columns
data[numeric_columns].boxplot(figsize=(8, 6))
plt.title("Global Boxplots for All Numeric Variables")
plt.tight_layout()
plt.show()

# Boxplots for each variable

def plot_combined_boxplots(data, numeric_columns, file_tag):
    num_plots = len(numeric_columns)
    num_cols = 3  # Number of columns in the grid
    num_rows = (num_plots + num_cols - 1) // num_cols  # Calculate the number of rows needed

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 5))
    axes = axes.flatten()  # Flatten the axes array for easy iteration

    for i, column in enumerate(numeric_columns):
        axes[i].boxplot(data[column].dropna())
        axes[i].set_title(f'Boxplot for {column}')
        axes[i].set_xlabel(column)
        axes[i].set_ylabel('Values')

    # Remove any empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(f"projeto/charts/all_boxplots_{file_tag}.png")
    plt.show()

plot_combined_boxplots(data, numeric_columns, file_tag)

# Histograms for each variable

def plot_combined_histograms(data, numeric_columns, file_tag):
    num_plots = len(numeric_columns)
    num_cols = 3  # Number of columns in the grid
    num_rows = (num_plots + num_cols - 1) // num_cols  # Calculate the number of rows needed

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 5))
    axes = axes.flatten()  # Flatten the axes array for easy iteration

    for i, column in enumerate(numeric_columns):
        data[column].hist(bins=30, color='skyblue', edgecolor='black', ax=axes[i])
        axes[i].set_title(f"Histogram for {column}")
        axes[i].set_xlabel(column)
        axes[i].set_ylabel("Frequency")

    # Remove any empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(f"projeto/charts/all_histograms_{file_tag}.png")
    plt.show()

plot_combined_histograms(data, numeric_columns, file_tag)

# Number of outliers per variable

def count_outliers(column):
     Q1 = column.quantile(0.25)
     Q3 = column.quantile(0.75)
     IQR = Q3 - Q1
     lower_bound = Q1 - 1.5 * IQR
     upper_bound = Q3 + 1.5 * IQR
    
    
     outliers = column[(column < lower_bound) | (column > upper_bound)]
     return len(outliers)  # Return the number of outliers


outlier_count = {column: count_outliers(data[column]) for column in numeric_columns}

def plot_outliers_count(outliers_count, title="Number of Outliers per Variable"):
    labels = list(outliers_count.keys())
    values = list(outliers_count.values())

    plt.figure(figsize=(8, 5))
    plt.bar(labels, values, color='skyblue')
    plt.title(title)
    plt.xlabel("Variables")
    plt.ylabel("Number of Outliers")
    
   
    for i, value in enumerate(values):
        plt.text(i, value + 0.1, f'{value}', ha='center', va='bottom', fontsize=10)

    plt.xticks(rotation=90)  
    plt.tight_layout()  
    plt.show()

plot_outliers_count(outlier_count)

# Compute known distributions
# def compute_known_distributions(x_values: list) -> dict:
#     distributions = dict()

#     # Gaussian (Normal) Distribution
#     mean, sigma = norm.fit(x_values)
#     distributions["Normal(%.1f, %.2f)" % (mean, sigma)] = norm.pdf(np.linspace(min(x_values), max(x_values), 1000), mean, sigma)

#     # Exponential Distribution
#     loc, scale = expon.fit(x_values)
#     distributions["Exp(%.2f)" % (1 / scale)] = expon.pdf(np.linspace(min(x_values), max(x_values), 1000), loc, scale)

#     # Log-Normal Distribution
#     sigma, loc, scale = lognorm.fit(x_values)
#     distributions["LogNor(%.1f, %.2f)" % (np.log(scale), sigma)] = lognorm.pdf(np.linspace(min(x_values), max(x_values), 1000), sigma, loc, scale)

#     return distributions


# def histogram_with_distributions(ax, series, var):
#     values = series.sort_values().dropna().to_list()

#     # Plot histogram
#     ax.hist(values, density=True, color='skyblue', edgecolor='black')  
#     distributions = compute_known_distributions(values)

#     # Plot distribution fits
#     for label, distribution in distributions.items():
#         ax.plot(np.linspace(min(values), max(values), 1000), distribution, label=label, linestyle='-', linewidth=2)
    
#     ax.set_title(f"Best fit for {var}")
#     ax.set_xlabel(var)
#     ax.set_ylabel("Density")
#     ax.legend()


# # Example of plotting for all numeric columns
# numeric_columns = data.select_dtypes(include=["number"]).columns
# if len(numeric_columns) > 0:
#     for column in numeric_columns:
#         # Create a new figure for each histogram
#         plt.figure(figsize=(8, 5))  # Adjust the size of each plot
#         histogram_with_distributions(plt.gca(), data[column], column)  # Use current axes (gca) for each plot
#         plt.tight_layout()  # Ensure the layout is not cropped
#         plt.show()  # Show each plot individually
# else:
#     print("No numeric columns available.")

# Ensure the target column exists in the dataset
if "LAW_CAT_CD" in data.columns:
    # Count the number of occurrences of each class
    class_distribution = data["LAW_CAT_CD"].value_counts()

    # Plot the distribution
    plt.figure(figsize=(8, 6))
    class_distribution.plot(kind='bar', color='skyblue', edgecolor='black')

    # Add labels and title
    plt.title("Class Distribution of Target Variable 'LAW_CAT_CD'", fontsize=14)
    plt.xlabel("Class", fontsize=12)
    plt.ylabel("Number of Records", fontsize=12)

    # Annotate bar heights
    for index, value in enumerate(class_distribution):
        plt.text(index, value + 0.5, str(value), ha='center', fontsize=10)

    # Show the plot
    plt.tight_layout()
    plt.show()
else:
    print("The column 'LAW_CAT_CD' is not present in the dataset.")