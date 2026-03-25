"""
Compute expectancy for top-k percentiles of model predictions for baseline and rich models.
Writes `backtests/expectancy_by_percentile.csv` and prints summary.
"""
import os, joblib, pandas as pd, numpy as np
BASE=os.path.join(os.path.dirname(__file__),'..')
FEAT=os.path.join('datasets','engineered_features.parquet')
RICH=os.path.join('datasets','rich_features.parquet')
M1=os.path.join('models','lgbm_return_v1.pkl')
M2=os.path.join('models','lgbm_return_rich.pkl')
OUT=os.path.join('backtests','expectancy_by_percentile.csv')
os.makedirs(os.path.dirname(OUT),exist_ok=True)

# params
NOTIONAL=25.0
FEE=0.001
SLIP=0.001
TARGET_VOL=0.02

# load data
df=pd.read_parquet(FEAT).sort_values(['symbol','timestamp']).reset_index(drop=True)
if 'price_next' not in df.columns:
    df['price_next']=df.groupby('symbol')['price'].shift(-1)
df=df.dropna(subset=['price_next']).reset_index(drop=True)

# load models
m1=joblib.load(M1)
if os.path.exists(M2): m2=joblib.load(M2)
else: m2=m1

# preds
BASE_FEATS=['event_id','qty','price','fee_usd','price_change','price_roll3']
for c in BASE_FEATS:
    if c not in df.columns: df[c]=0.0
Xb=df[BASE_FEATS].fillna(0.0)
p1=m1.predict(Xb)
df['pred_base']=p1

if os.path.exists(RICH):
    dr=pd.read_parquet(RICH).sort_values(['symbol','timestamp']).reset_index(drop=True)
    RICH_FEATS=['qty','price','ret','rv_10','rv_10_ann','atr','vol_spike']
    for c in RICH_FEATS:
        if c not in dr.columns: dr[c]=0.0
    Xr=dr[RICH_FEATS].fillna(0.0)
    p2=m2.predict(Xr)
    dr['pred_rich']=p2
    # align by index length
    if len(dr)>=len(df):
        df['pred_rich']=dr['pred_rich'].values[:len(df)]
    else:
        # pad with baseline predictions if shorter
        pad=np.full(len(df)-len(dr), p2.mean())
        df['pred_rich']=np.concatenate([dr['pred_rich'].values, pad])
else:
    df['pred_rich']=df['pred_base']

# percentiles to evaluate (top X%)
percentiles = list(range(1,51,1))  # top 1%..50%
rows=[]
for pct in percentiles:
    # baseline: select top pct by pred
    cutoff_b = np.percentile(df['pred_base'], 100-pct)
    sel_b = df[df['pred_base']>=cutoff_b].copy()
    sel_b['gross_ret'] = (sel_b['price_next']-sel_b['price'])/sel_b['price']
    sel_b['pnl_net'] = sel_b['gross_ret']*NOTIONAL - (sel_b['fee_usd'].fillna(NOTIONAL*(FEE+SLIP)))
    n_b=len(sel_b)
    total_b = sel_b['pnl_net'].sum() if n_b>0 else 0.0

    # rich: top pct by pred_rich with vol sizing
    cutoff_r = np.percentile(df['pred_rich'], 100-pct)
    sel_r = df[df['pred_rich']>=cutoff_r].copy()
    pnls_r=[]
    for _,rr in sel_r.iterrows():
        vol_ann = float(rr.get('rv_10_ann') or 0.0)
        entry=float(rr['price']); exit_p=float(rr['price_next'])
        gross=(exit_p-entry)/entry
        if vol_ann<=0: continue
        target_dollar_vol = TARGET_VOL * max(1.0, getattr(__import__('config'),'INITIAL_CAPITAL',100.0))
        notional = target_dollar_vol/vol_ann
        if (notional*entry) < getattr(__import__('config'),'MIN_ORDER_VALUE',0.0): continue
        pnl = gross*notional - (FEE+SLIP)*notional
        pnls_r.append(pnl)
    n_r=len(pnls_r); total_r=sum(pnls_r)
    rows.append({'top_pct':pct,'n_base':n_b,'total_pnl_base':total_b,'n_rich':n_r,'total_pnl_rich':total_r})

out=pd.DataFrame(rows)
out.to_csv(OUT,index=False)
print('Wrote',OUT)
print(out.head(15).to_string(index=False))
