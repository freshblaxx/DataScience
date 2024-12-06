from pandas import read_csv, DataFrame, Series, to_numeric
import pandas as pd

NR_STDEV: int = 2

def get_variable_types(df: DataFrame) -> dict[str, list]:
    """Classifies columns as 'numeric' or 'binary'."""
    variable_types: dict = {"numeric": [], "binary": []}
    nr_values: Series = df.nunique(axis=0, dropna=True)

    for c in df.columns:
        if nr_values[c] == 2 and set(df[c].dropna().unique()).issubset({0, 1}):
            variable_types["binary"].append(c)
            df[c] = df[c].astype("bool")
        else:
            try:
                to_numeric(df[c], errors="raise")
                variable_types["numeric"].append(c)
            except ValueError:
                pass  # Ignore columns that cannot be converted to numeric

    return variable_types

def determine_outlier_thresholds_for_var(
    summary5: Series, std_based: bool = True, threshold: float = NR_STDEV
) -> tuple[float, float]:
    """Determines the upper and lower outlier thresholds."""
    if std_based:
        std = threshold * summary5["std"]
        top = summary5["mean"] + std
        bottom = summary5["mean"] - std
    else:
        top = summary5["75%"]
        bottom = summary5["25%"]
    return bottom, top

def drop_outliers(
    df: DataFrame, variable_types: dict[str, list], exclude_columns: list[str] = []
) -> tuple[DataFrame, dict[str, int]]:
    """Drops outliers from specified columns but retains excluded columns."""
    outliers_info = {}
    mask = pd.Series(True, index=df.index)  # Start with a mask that includes all rows

    for col in variable_types["numeric"]:
        if col in exclude_columns:
            continue
        summary = df[col].describe()
        bottom, top = determine_outlier_thresholds_for_var(summary)
        outliers_count = df[(df[col] < bottom) | (df[col] > top)].shape[0]
        outliers_info[col] = outliers_count
        mask &= (df[col] >= bottom) & (df[col] <= top)  # Update mask for non-excluded columns

    filtered_df = df[mask]  # Apply the mask to drop outliers only in non-excluded columns
    return filtered_df, outliers_info

# Example usage
data: DataFrame = read_csv("/Users/tomifemme/Desktop/DataScience/Projeto/Preparation/Outliers/data_cleaned.csv")

variable_types = get_variable_types(data)
print("Variable types:", variable_types)

cleaned_data, outliers_info = drop_outliers(
    data, variable_types, exclude_columns=["PERP_RACE", "AGE_GROUP", "ARREST_BORO"]
)

# Print outliers information
for col, count in outliers_info.items():
    print(f"Column '{col}': {count} outliers dropped")

# Save the cleaned data to a new CSV file
cleaned_data.to_csv("Outliers_dropped_arrests.csv", index=False)
print(f"Cleaned data saved to 'Outliers_dropped_arrests.csv'")
