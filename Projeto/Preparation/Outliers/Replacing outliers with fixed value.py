from pandas import read_csv, DataFrame, Series, to_numeric

NR_STDEV: int = 2

def get_variable_types(df: DataFrame) -> dict[str, list]:
    variable_types: dict = {"numeric": [], "binary": []}

    nr_values: Series = df.nunique(axis=0, dropna=True)
    for c in df.columns:
        if 2 == nr_values[c] and set(df[c].dropna().unique()).issubset({0, 1}):
            variable_types["binary"].append(c)
            df[c] = df[c].astype("bool")
        else:
            to_numeric(df[c], errors="raise")
            variable_types["numeric"].append(c)

    return variable_types

def determine_outlier_thresholds_for_var(
    summary5: Series, std_based: bool = True, threshold: float = NR_STDEV
) -> tuple[float, float]:
    if std_based:
        std: float = threshold * summary5["std"]
        top = summary5["mean"] + std
        bottom = summary5["mean"] - std
    else:
        top = summary5["75%"]
        bottom = summary5["25%"]
    return bottom, top

def replace_outliers_with_median(df: DataFrame, variable_types: dict[str, list]) -> DataFrame:
    outliers_info = {}
    for col in variable_types["numeric"]:
        summary = df[col].describe()
        bottom, top = determine_outlier_thresholds_for_var(summary)
        median: float = df[col].median()

        # Replace outliers with the median
        outliers_count = ((df[col] < bottom) | (df[col] > top)).sum()
        outliers_info[col] = outliers_count
        df[col] = df[col].apply(lambda x: median if x > top or x < bottom else x)

    return df, outliers_info

# Example usage
data: DataFrame = read_csv("/Users/tomifemme/Desktop/DataScience/Projeto/Preparation/Outliers/data_cleaned.csv")

variable_types = get_variable_types(data)
print("Variable Types:", variable_types)

cleaned_data, outliers_info = replace_outliers_with_median(data, variable_types)

# Print outliers information
for col, count in outliers_info.items():
    print(f"Column '{col}': {count} outliers replaced out of {len(data)} total entries")

# Save the cleaned data to a new CSV file
cleaned_data.to_csv("Outliers_replaced_arrests.csv", index=False)
print(f"Cleaned data saved to 'Outliers_replaced_arrests.csv'")
