from pandas import read_csv, DataFrame, Series
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from matplotlib.pyplot import subplots, show


file = "Financial"
data: DataFrame = read_csv("Projeto\Preparation\Outliers\Financial_Alt1_truncate_outliers.csv", na_values="")
target = "CLASS"
vars: list[str] = data.columns.to_list()
target_data: Series = data.pop(target)


# Standard Scaler

transf: StandardScaler = StandardScaler(with_mean=True, with_std=True, copy=True).fit(
    data
)
df_zscore = DataFrame(transf.transform(data), index=data.index)
df_zscore[target] = target_data
df_zscore.columns = vars
df_zscore.to_csv(f"Projeto\Preparation\Scaling/{file}_scaled_zscore.csv", index="id")


# MinMax Scaler
transf: MinMaxScaler = MinMaxScaler(feature_range=(0, 1), copy=True).fit(data)
df_minmax = DataFrame(transf.transform(data), index=data.index)
df_minmax[target] = target_data
df_minmax.columns = vars
df_minmax.to_csv(f"Projeto\Preparation\Scaling/{file}_scaled_minmax.csv", index="id")

# Plots

fig, axs = subplots(1, 3, figsize=(20, 10), squeeze=False)
axs[0, 1].set_title("Original data")
data.boxplot(ax=axs[0, 0])
axs[0, 0].set_title("Z-score normalization")
df_zscore.boxplot(ax=axs[0, 1])
axs[0, 2].set_title("MinMax normalization")
df_minmax.boxplot(ax=axs[0, 2])
show()