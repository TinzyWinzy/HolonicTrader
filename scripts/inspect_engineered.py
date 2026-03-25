import pandas as pd
p='datasets/engineered_features.parquet'
df=pd.read_parquet(p)
print('columns=',list(df.columns))
print(df.head(5))
