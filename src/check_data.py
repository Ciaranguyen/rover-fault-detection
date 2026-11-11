"""E0: kiểm tra nhanh dữ liệu sinh ra.
In phân bố |phần dư range| = |r_ik - ||p_hat_i - p_hat_k||| cho từng nhóm:
cặp trung thực (đầu và cuối episode), rover trôi, rover khai giả, và spoof khớp range một phía
(rover nhận j* so với các rover khác).

Chạy: python src/check_data.py data/val.npz
"""
import numpy as np, json, sys
d=np.load(sys.argv[1]); meta=json.load(open(sys.argv[1].replace('.npz','_meta.json')))['episodes']
ph,r,lab=d['p_hat'],d['r'],d['label']
E,T,N,_=ph.shape
est=np.linalg.norm(ph[:,:,:,None,:]-ph[:,:,None,:,:],axis=-1)
res=np.abs(r-est)  # (E,T,N,N) |residual| seen by receiver i about k (symmetric)
def summary(mask,name):
    v=res[mask]; v=v[np.isfinite(v)]
    print(f'{name:28s} n={len(v):7d} median {np.median(v):.3f}  p90 {np.percentile(v,90):.3f}  p99 {np.percentile(v,99):.3f}')
# honest pairs (both honest at t)
lab_i=lab[:,:,:,None]; lab_k=lab[:,:,None,:]
summary((lab_i==0)&(lab_k==0)&(np.arange(T)[None,:,None,None]<150),'honest pair, t<150')
summary((lab_i==0)&(lab_k==0)&(np.arange(T)[None,:,None,None]>=300),'honest pair, t>=300')
summary((lab_i==0)&(lab_k==1),'about drift rover')
summary((lab_i==0)&(lab_k==2),'about spoof rover (all)')
# spoof_one: receiver j* vs others, after 40 steps
for e,m in enumerate(meta):
    pass
mj=np.zeros_like(res,bool); mo=np.zeros_like(res,bool)
for e,m in enumerate(meta):
    if m['fault_type']=='spoof_one':
        k=m['faulty'][0]; j=m['spoof_receiver']; t0=m['t_u']+40
        mj[e,t0:,j,k]=True
        others=[i for i in range(N) if i not in (j,k)]
        mo[e,t0:,others,k]=True
summary(mj,'spoof_one: receiver j*')
summary(mo,'spoof_one: other receivers')
print('neighbors in range per rover per step: mean %.2f'%(np.isfinite(r).sum(-1).mean()))
