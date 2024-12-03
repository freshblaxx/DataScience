from pandas import read_csv, DataFrame, Series
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from matplotlib.pyplot import subplots, show
from matplotlib.pyplot import gca,savefig


file = "arrests"
data: DataFrame = read_csv("/Users/tomifemme/Desktop/DataScience/Projeto/Preparation/Scaling/Outliers_replaced_arrests.csv")
target = "CLASS"
target_data: Series = data.pop(target)

# Standard Scaler
transf: StandardScaler = StandardScaler(with_mean=True, with_std=True).fit(data)
df_zscore = DataFrame(transf.transform(data), index=data.index, columns=data.columns)
df_zscore[target] = target_data
df_zscore.to_csv(f"/Users/tomifemme/Desktop/DataScience/Projeto/Preparation/Scaling/{file}_scaled_zscore_arrest.csv", index=False)

# MinMax Scaler
transf: MinMaxScaler = MinMaxScaler(feature_range=(0, 1)).fit(data)
df_minmax = DataFrame(transf.transform(data), index=data.index, columns=data.columns)
df_minmax[target] = target_data
df_minmax.to_csv(f"/Users/tomifemme/Desktop/DataScience/Projeto/Preparation/Scaling/{file}_scaled_minmax_arrest.csv", index=False)

# Plots

#fig, axs = subplots(1, 3, figsize=(20, 10), squeeze=False)
#axs[0, 1].set_title("Original data")
#data.boxplot(ax=axs[0, 0])
#axs[0, 0].set_title("Z-score normalization")
#df_zscore.boxplot(ax=axs[0, 1])
#axs[0, 2].set_title("MinMax normalization")
#df_minmax.boxplot(ax=axs[0, 2])
#savefig(f"Projeto/Images/Scaling_Plots.png")
#show()