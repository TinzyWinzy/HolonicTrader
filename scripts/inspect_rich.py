import pandas as pd
p='datasets/rich_features.parquet'
print('reading',p)
df=pd.read_parquet(p)
print('columns=',list(df.columns))
print(df.head(5))
