from pandas import read_csv, DataFrame, Series
from sklearn.preprocessing import StandardScaler

file = "Ny_Arrest"
data: DataFrame = read_csv("Projeto/Preparation/new_class_ny_arrests.csv", na_values="")
target = "LAW_CAT_CD"
vars: list[str] = data.columns.to_list()
target_data: Series = data.pop(target)

transf: StandardScaler = StandardScaler(with_mean=True, with_std=True, copy=True).fit(
    data
)
df_zscore = DataFrame(transf.transform(data), index=data.index)
df_zscore[target] = target_data
df_zscore.columns = vars
df_zscore.to_csv(f"Projeto\Preparation\Scaling/{file}_scaled_zscore.csv", index="id")