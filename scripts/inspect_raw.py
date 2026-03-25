import pandas as pd
p='datasets/raw_trades_snapshot.parquet'
df=pd.read_parquet(p)
print('columns=',list(df.columns))
print(df.head(5))
