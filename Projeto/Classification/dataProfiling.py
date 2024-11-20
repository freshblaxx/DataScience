from pandas import read_csv, DataFrame, Series
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig, show, subplots
from scipy.stats import norm, expon, lognorm
import numpy as np
import pandas as pd



filename = "class_financial distress.csv"
file_tag = "financial"

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

def plot_individual_boxplots(data, numeric_columns):
     for column in numeric_columns:
         plt.figure(figsize=(6, 4)) 
         plt.boxplot(data[column].dropna()) 
         plt.title(f'Boxplot for {column}')  
         plt.xlabel(column)  
         plt.ylabel('Values')  
         plt.tight_layout() 
         plt.show() 

plot_individual_boxplots(data, numeric_columns)

# Histograms for each variable

def plot_histograms(data, numeric_columns):
     for column in numeric_columns:
         plt.figure(figsize=(6, 4)) 
         data[column].hist(bins=30, color='skyblue', edgecolor='black')  
         plt.title(f"Histogram for {column}")
         plt.xlabel(column)
         plt.ylabel("Frequency")
         plt.tight_layout()  
         plt.show()

plot_histograms(data, numeric_columns)

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
def compute_known_distributions(x_values: list) -> dict:
    distributions = dict()

    # Gaussian (Normal) Distribution
    mean, sigma = norm.fit(x_values)
    distributions["Normal(%.1f, %.2f)" % (mean, sigma)] = norm.pdf(np.linspace(min(x_values), max(x_values), 1000), mean, sigma)

    # Exponential Distribution
    loc, scale = expon.fit(x_values)
    distributions["Exp(%.2f)" % (1 / scale)] = expon.pdf(np.linspace(min(x_values), max(x_values), 1000), loc, scale)

    # Log-Normal Distribution
    sigma, loc, scale = lognorm.fit(x_values)
    distributions["LogNor(%.1f, %.2f)" % (np.log(scale), sigma)] = lognorm.pdf(np.linspace(min(x_values), max(x_values), 1000), sigma, loc, scale)

    return distributions


def histogram_with_distributions(ax, series, var):
    values = series.sort_values().dropna().to_list()

    # Plot histogram
    ax.hist(values, density=True, color='skyblue', edgecolor='black')  
    distributions = compute_known_distributions(values)

    # Plot distribution fits
    for label, distribution in distributions.items():
        ax.plot(np.linspace(min(values), max(values), 1000), distribution, label=label, linestyle='-', linewidth=2)
    
    ax.set_title(f"Best fit for {var}")
    ax.set_xlabel(var)
    ax.set_ylabel("Density")
    ax.legend()


# Example of plotting for all numeric columns
numeric_columns = data.select_dtypes(include=["number"]).columns
if len(numeric_columns) > 0:
    for column in numeric_columns:
        # Create a new figure for each histogram
        plt.figure(figsize=(8, 5))  # Adjust the size of each plot
        histogram_with_distributions(plt.gca(), data[column], column)  # Use current axes (gca) for each plot
        plt.tight_layout()  # Ensure the layout is not cropped
        plt.show()  # Show each plot individually
else:
    print("No numeric columns available.")