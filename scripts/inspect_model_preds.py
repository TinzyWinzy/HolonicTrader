import joblib, pandas as pd, os, numpy as np
FEAT=os.path.join('datasets','engineered_features.parquet')
RICH=os.path.join('datasets','rich_features.parquet')
M1=os.path.join('models','lgbm_return_v1.pkl')
M2=os.path.join('models','lgbm_return_rich.pkl')

df=pd.read_parquet(FEAT).sort_values(['symbol','timestamp']).reset_index(drop=True)
if 'price_next' not in df.columns:
    df['price_next']=df.groupby('symbol')['price'].shift(-1)
X_base_cols=['event_id','qty','price','fee_usd','price_change','price_roll3']
for c in X_base_cols:
    if c not in df.columns: df[c]=0.0
X_base=df[X_base_cols].fillna(0.0)

m1=joblib.load(M1)
preds1=m1.predict(X_base)
print('Baseline preds stats: min,1%,5%,25%,50%,75%,95%,99%,max')
print(np.percentile(preds1,[0,1,5,25,50,75,95,99,100]))

if os.path.exists(RICH):
    df2=pd.read_parquet(RICH).sort_values(['symbol','timestamp']).reset_index(drop=True)
    X_r_cols=['qty','price','ret','rv_10','rv_10_ann','atr','vol_spike']
    for c in X_r_cols:
        if c not in df2.columns: df2[c]=0.0
    m2=joblib.load(M2) if os.path.exists(M2) else m1
    X_r=df2[X_r_cols].fillna(0.0)
    preds2=m2.predict(X_r)
    print('Rich preds stats:')
    print(np.percentile(preds2,[0,1,5,25,50,75,95,99,100]))
else:
    print('No rich features file')
