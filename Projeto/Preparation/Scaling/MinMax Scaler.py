from pandas import read_csv, DataFrame, Series
from sklearn.preprocessing import MinMaxScaler

file = "Financial"
data: DataFrame = read_csv("Projeto/Preparation/class_financial distress.csv", na_values="")
target = "CLASS"
vars: list[str] = data.columns.to_list()
target_data: Series = data.pop(target)


transf: MinMaxScaler = MinMaxScaler(feature_range=(0, 1), copy=True).fit(data)
df_minmax = DataFrame(transf.transform(data), index=data.index)
df_minmax[target] = target_data
df_minmax.columns = vars
df_minmax.to_csv(f"Projeto\Preparation\Scaling/{file}_scaled_minmax.csv", index="id")