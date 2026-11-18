from pathlib import Path
import json, sys
import numpy as np

root=Path(sys.argv[1])
out=Path(__file__).parent
w=np.array([.299,.587,.114])
def chroma(y):
    return y-(y@w)[:,None]*w/(w@w)
def mlp(x,y,xt,seed,path):
    rng=np.random.default_rng(seed)
    p=[rng.normal(0,np.sqrt(2/x.shape[1]),(x.shape[1],32)),np.zeros(32),
       rng.normal(0,.02,(32,3)),np.zeros(3)]
    m=[np.zeros_like(t) for t in p]; v=[np.zeros_like(t) for t in p]
    step=0
    for ep in range(120):
        for ids in np.array_split(rng.permutation(len(x)),int(np.ceil(len(x)/128))):
            a=x[ids]; h=np.tanh(a@p[0]+p[1]); z=(h@p[2]+p[3]-y[ids])*(2/(3*len(ids)))
            dh=(z@p[2].T)*(1-h*h)
            grad=[a.T@dh,dh.sum(0),h.T@z,z.sum(0)]
            step+=1
            for j,g in enumerate(grad):
                m[j]=.9*m[j]+.1*g; v[j]=.999*v[j]+.001*g*g
                p[j]-=.001*m[j]/(1-.9**step)/(np.sqrt(v[j]/(1-.999**step))+1e-8)
    np.savez(path,*p)
    return np.tanh(xt@p[0]+p[1])@p[2]+p[3]
def summary(pred,t,base,scenes):
    n=np.linalg.norm(pred,axis=1); u=np.linalg.norm(t,axis=1)
    delta=(2*(pred*t).sum(1)-(pred*pred).sum(1))/12
    assert (base-delta>0).all()
    d=10*np.log10(base/(base-delta))
    return dict(offset_rmse=float(np.sqrt(np.mean((pred-t)**2))/2),
      cosine_mean=float(np.mean((pred*t).sum(1)/np.maximum(n*u,1e-15))),
      offset_norm_ratio=float(n.mean()/u.mean()),
      preclip_psnr_db=float(np.mean([d[scenes==s].mean() for s in np.unique(scenes)])))
result=[]
for seed in (56,57,58):
    files=sorted(root.glob(f'09_*/remote_snapshots/*/features/source_v3_seed{seed}.npz'))
    q=np.load(files[0]); x=q['features'].astype(float); y=chroma(q['targets'].astype(float))
    scenes=q['scenes']; u=np.unique(scenes)
    test=np.isin(scenes,np.random.default_rng(20261115).permutation(u)[:len(u)//5]); fit=~test
    assert np.isfinite(x).all() and np.isfinite(y).all() and not set(scenes[fit])&set(scenes[test])
    mu=x[fit].mean(0); sd=np.maximum(1e-6,x[fit].std(0)); x=np.clip((x-mu)/sd,-10,10)
    scale=np.sqrt(np.mean(y[fit]**2)); a=np.column_stack([x,np.ones(len(x))])
    ridge=np.linalg.solve(a[fit].T@a[fit]+np.diag([len(x[fit])*1e-3]*x.shape[1]+[0]),a[fit].T@y[fit])
    pred=chroma(mlp(x[fit],y[fit]/scale,x,seed,out/f'mlp_{seed}.npz')*scale)
    np.savez(out/f'preprocessing_{seed}.npz',mu=mu,sd=sd,scale=scale,ridge=ridge,features=q['feature_names'],train_scenes=np.unique(scenes[fit]),holdout_scenes=np.unique(scenes[test]))
    values=dict(seed=seed,source_file=str(files[0]),fit_crops=int(fit.sum()),holdout_crops=int(test.sum()))
    for key,mask in [('fit',fit),('holdout',test)]:
        values[key]={name:summary(z[mask],y[mask],q['base_mse'][mask],scenes[mask])
          for name,z in [('no_correction',np.zeros_like(y)),('ridge',chroma(a@ridge)),('mlp',pred),('oracle',y)]}
    result.append(values)
    print(json.dumps(values),flush=True)
(out/'source_results.json').write_text(json.dumps(result,indent=2)+'\n')
