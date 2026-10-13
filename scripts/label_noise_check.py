"""Label noise vs gold-58 (E016): gold positive rate by blend cell status, and gold AUC under other silence rules.
    python scripts/label_noise_check.py"""
import numpy as np, pandas as pd
from knee.labels import public_sets as PS
from knee.labels.blend import blend, BlendConfig
from knee.labels.build_labels import gold_frame
from knee.paths import LABELS, load_paths
from sklearn.metrics import roc_auc_score
from knee.utils.metrics import per_finding_auc
P=load_paths(); train=pd.read_csv(P.train_csv); uids=train.StudyInstanceUID.tolist(); gold=gold_frame(train)
root=PS.public_dir(P.cache_root); frames={n:PS.load(n,root) for n in PS.DEFAULT_READERS}
p,w,st=blend(frames,uids,BlendConfig())
g=gold.index; pg,sg=p.loc[g],st.loc[g]
rows=[]
for k in LABELS:
    y=gold[k]; r={'finding':k,'gold_pos':int(y.sum())}
    for s in ('known','fill','silent'):
        m=sg[k]==s; r[f'{s}_n']=int(m.sum()); r[f'{s}_pos%']=round(100*y[m].mean()) if m.any() else None
    band=(pg[k]>0.4)&(pg[k]<0.6); r['band_n']=int(band.sum()); r['band_pos%']=round(100*y[band].mean()) if band.any() else None
    b=(pg[k]>=0.6).astype(float).where(~band); ok=b.notna()
    r['hard_err']=int((b[ok]!=y[ok]).sum()); r['hard_n']=int(ok.sum())
    r['auc_soft']=round(roc_auc_score(y,pg[k]),3)
    pz=pg[k].where(sg[k]!='silent',0.0); r['auc_silent=0']=round(roc_auc_score(y,pz),3)
    rows.append(r)
d=pd.DataFrame(rows).set_index('finding'); print(d.to_string())
# corpus-wide: share of non-gold cells in each status and in the masked eval band
ng=[u for u in uids if u not in g]
print('\ncorpus: share masked in val (0.4<p<0.6) / silent / fill')
print(pd.DataFrame({'band':((p.loc[ng]>0.4)&(p.loc[ng]<0.6)).mean(),'silent':(st.loc[ng]=='silent').mean(),'fill':(st.loc[ng]=='fill').mean()}).round(3).to_string())

# ---- silence-rule sweep
res={}
for name,cfg in {'v0 (silent 0.5, vote)':BlendConfig(),
 'silent cells only -> 0.15':None,
 'silent_p 0.3 everywhere':BlendConfig(silent_p=0.3),
 'silent_p 0.2 everywhere':BlendConfig(silent_p=0.2),
 'no fills':BlendConfig(fills={}),
 'drop lixin':'lixin'}.items():
    if cfg is None:
        p,w,st=blend(frames,uids,BlendConfig()); p=p.mask(st=='silent',0.15)
    elif isinstance(cfg,str):
        p,_,_=blend({k:v for k,v in frames.items() if k!=cfg},uids,BlendConfig())
    else: p,_,_=blend(frames,uids,cfg)
    res[name]=per_finding_auc(gold.to_numpy(),p.loc[gold.index,LABELS].to_numpy())
d=pd.DataFrame(res,index=LABELS); d.loc['MACRO']=d.mean(); print(d.round(3).to_string())
