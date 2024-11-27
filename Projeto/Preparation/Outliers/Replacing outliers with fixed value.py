from pandas import read_csv, DataFrame, Series, to_numeric, to_datetime

# Define constants
NR_STDEV: int = 2

# Function to classify variable types
def get_variable_types(df: DataFrame) -> dict[str, list]:
    variable_types: dict = {"numeric": [], "binary": [], "date": [], "symbolic": []}

    nr_values: Series = df.nunique(axis=0, dropna=True)
    for c in df.columns:
        if 2 == nr_values[c]:
            variable_types["binary"].append(c)
            df[c].astype("bool")
        else:
            try:
                to_numeric(df[c], errors="raise")
                variable_types["numeric"].append(c)
            except ValueError:
                try:
                    df[c] = to_datetime(df[c], errors="raise")
                    variable_types["date"].append(c)
                except ValueError:
                    variable_types["symbolic"].append(c)

    return variable_types

# Function to determine outlier thresholds
def determine_outlier_thresholds_for_var(
    summary5: Series, std_based: bool = True, threshold: float = NR_STDEV
) -> tuple[float, float]:
    top: float = 0
    bottom: float = 0
    if std_based:
        std: float = threshold * summary5["std"]
        top = summary5["mean"] + std
        bottom = summary5["mean"] - std
    else:
        iqr: float = threshold * (summary5["75%"] - summary5["25%"])
        top = summary5["75%"] + iqr
        bottom = summary5["25%"] - iqr

    return top, bottom

# Load data
file_tag = "Financial"
data: DataFrame = read_csv(
    r"Projeto\Preparation\class_financial distress.csv", na_values="", parse_dates=True, dayfirst=True
)
print(f"Original data: {data.shape}")

# Get numeric variables
numeric_vars: list[str] = get_variable_types(data)["numeric"]

if [] != numeric_vars:
    df: DataFrame = data.copy(deep=True)
    summary5 = data.describe()  # Compute summary statistics for numeric variables
    for var in numeric_vars:
        top, bottom = determine_outlier_thresholds_for_var(summary5[var])  # Pass summary statistics
        median: float = df[var].median()
        df[var] = df[var].apply(lambda x: median if x > top or x < bottom else x)
    
    # Save cleaned data
    df.to_csv(f"Projeto\Preparation\Outliers/{file_tag}_replacing_outliers.csv", index=True)
    print("Data after replacing outliers:", df.shape)
    print(df.describe())
else:
    print("There are no numeric variables")
