from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import hashlib
import json
import sys
import time
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from threadpoolctl import threadpool_limits

STUDY=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(STUDY/'analysis'))
from physics import TIMES,T2_GRID,pore_lookup
from inversion import INVERSE_T2,INVERSE_EDGES
from design import DATA,MOMENTS

STARTS=np.array([0.,3.,6.,24.])


@lru_cache(maxsize=1)
def operator():
    model=np.load(STUDY/'data/dephasing_model.npz')
    obs=np.load(STUDY/'data/observation_kernel_full_100.npz')
    kernels=model['offset_kernel_nV']
    weights=model['frequency_weights_5']
    effective=np.einsum('oqzk,to->qtzk',kernels,weights,optimize=True)
    effective*=np.exp(-TIMES[None,:,None,None]/T2_GRID)
    bg=interp1d(np.log(T2_GRID),kernels[:,:,40:,:].sum(axis=2),axis=-1)(np.log(.15))
    background=.25*np.einsum('oq,to->qt',bg,weights)*np.exp(-TIMES/.15)
    return effective[:,:,:40,:],background,obs['q_ams']


@lru_cache(maxsize=1)
def inverse_operator():
    model=np.load(STUDY/'data/dephasing_model.npz')
    obs=np.load(STUDY/'data/observation_kernel_full_100.npz')
    z=obs['z_m']; kernels=model['offset_kernel_nV']
    groups=[kernels[:,:,(z>=lo)&(z<hi),:].sum(axis=2)
            for lo,hi in zip(INVERSE_EDGES[:-1],INVERSE_EDGES[1:])]
    grouped=np.stack(groups,axis=2)
    grouped=interp1d(np.log(T2_GRID),grouped,axis=-1)(np.log(INVERSE_T2))
    a=np.einsum('oqzk,to->qtzk',grouped,model['frequency_weights_5'],optimize=True)
    a*=np.exp(-TIMES[None,:,None,None]/INVERSE_T2)
    return a


@lru_cache(maxsize=32)
def lookup(alpha,n,rho):
    return pore_lookup(alpha,n,rho)


def actual_parameters(row,tag,stored):
    path=STUDY/f'work/physical_recovery_20260929/hydrus/{tag}/{row["case_id"]}/SELECTOR.IN'
    if not path.exists():return stored
    lines=path.read_text().splitlines()
    i=next(i for i,s in enumerate(lines) if 'thr' in s and 'ths' in s and 'Alfa' in s)+1
    vals=np.array([[float(v) for v in line.split()[:5]] for line in lines[i:i+2]])
    return vals if int(row['order'])==0 else vals[::-1]


def convert(h,row,tag,common_rho=False):
    params=actual_parameters(row,tag,h['parameters'])
    theta=h['theta_node']; mat=h['material_node']; z=h['z_m']
    fine=np.zeros((*theta.shape,len(T2_GRID)))
    residual=np.zeros(len(z))
    for soil in (0,1):
        tr,ts,alpha,n,ks=params[soil]
        rho=7e-6 if common_rho else h['rho_m_per_s'][soil]
        se,table,diag=lookup(float(alpha),float(n),float(rho))
        inside=mat==soil
        saturation=np.clip((theta[:,inside]-tr)/(ts-tr),0,1)
        fine[:,inside,:]=(ts-tr)*interp1d(se,table,axis=0)(saturation)
        residual[inside]=tr
    cells=len(z)-1
    assert cells%40==0
    distribution=((fine[:,:-1]+fine[:,1:])/2).reshape(len(theta),40,cells//40,len(T2_GRID)).mean(axis=2)
    total=((theta[:,:-1]+theta[:,1:])/2).reshape(len(theta),40,cells//40).mean(axis=2)
    res=((residual[:-1]+residual[1:])/2).reshape(40,cells//40).mean(axis=1)
    capillary=total-res
    conservation=float(abs(distribution.sum(axis=-1)-capillary).max())
    if conservation>1e-4:raise ValueError(f'Pore-water conservation error {conservation}')
    return distribution,total,capillary,conservation,params


def interpolation_weights(saved,queries):
    queries=np.asarray(queries)
    assert queries.min()>=saved.min() and queries.max()<=saved.max()
    lo=np.clip(np.searchsorted(saved,queries,side='right')-1,0,len(saved)-2)
    frac=(queries-saved[lo])/(saved[lo+1]-saved[lo])
    weights=np.zeros((len(queries),len(saved)))
    weights[np.arange(len(queries)),lo]=1-frac
    weights[np.arange(len(queries)),lo+1]=frac
    return weights


@lru_cache(maxsize=128)
def acquisition_weights(saved_tuple,nq):
    saved=np.array(saved_tuple)
    repeats=192//nq
    query=7200+STARTS[:,None,None]*3600+16*(np.arange(repeats)[None,None,:]*nq+np.arange(nq)[None,:,None])
    w=interpolation_weights(saved,query.ravel()).reshape(len(STARTS),nq,repeats,len(saved)).mean(axis=2)
    assert np.allclose(w.sum(axis=-1),1.)
    return w


def synthesize_case(row,tag='standard',common_rho=False,wait=False):
    threadpool_limits(1)
    case=row['case_id']
    target=DATA/'signals'/tag/(case+('_commonrho' if common_rho else '')+'.npz')
    target.parent.mkdir(exist_ok=True,parents=True)
    source=DATA/'hydraulics'/tag/(case+'.npz')
    status=source.with_suffix('.json')
    start=time.monotonic()
    while wait and (not source.exists() or not status.exists()):
        if status.exists() and not json.loads(status.read_text()).get('accepted'):
            raise RuntimeError(f'Unaccepted hydraulic case {case}')
        if time.monotonic()-start>7200:raise TimeoutError(case)
        time.sleep(2)
    report=json.loads(status.read_text())
    if not report['accepted']:raise RuntimeError(f'Unaccepted hydraulic case {case}')
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    if target.exists():
        with np.load(target) as saved:
            if str(saved['hydraulic_sha256'])==digest:return case
    h=np.load(source)
    distribution,total,capillary,error,params=convert(h,row,tag,common_rho)
    a,bg,q=operator()
    arrays=dict(hydraulic_sha256=digest,parameters=params,
                starts_h=STARTS,max_conservation_error=error,
                storage_72h_mm=float(h['storage_m'][np.argmin(abs(h['time_s']-266400))]*1000),
                storage_24h_mm=float(h['storage_m'][np.argmin(abs(h['time_s']-93600))]*1000))
    for name,moments in MOMENTS.items():
        w=acquisition_weights(tuple(h['time_s']),len(moments))
        averaged=np.einsum('wqi,izk->wqzk',w,distribution,optimize=True)
        signal=np.einsum('wqzk,qtzk->wqt',averaged,a[moments],optimize=True)+bg[moments]
        arrays['signal_'+name]=signal.astype(np.complex64)
        arrays['capillary_'+name]=np.einsum('wqi,iz->wz',w/len(moments),capillary).reshape(4,20,2).mean(axis=-1)
    reference_positions=[np.argmin(abs(h['time_s']-(7200+v*3600))) for v in STARTS]
    arrays['snapshot_full']=np.einsum('wzk,qtzk->wqt',distribution[reference_positions],a,optimize=True)+bg
    if row.get('split')=='mechanism' or tag.startswith('pilot'):
        arrays.update(distribution=distribution[reference_positions].astype(np.float32),
                      theta_total=total[reference_positions].astype(np.float32),
                      theta_capillary=capillary[reference_positions].astype(np.float32),
                      t2_s=T2_GRID,initial_storage_mm=float(h['storage_m'][0]*1000))
    np.savez_compressed(target,**arrays)
    return case


def job(args):return synthesize_case(*args)


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',default=str(DATA/'manifest.csv'))
    p.add_argument('--split',default='library,development,mechanism');p.add_argument('--tag',default='standard')
    p.add_argument('--workers',type=int,default=2);p.add_argument('--common-rho',action='store_true')
    p.add_argument('--wait',action='store_true');a=p.parse_args()
    table=pd.read_csv(a.manifest)
    if 'split' in table:table=table[table.split.isin(a.split.split(','))]
    tasks=[(r,a.tag,a.common_rho,a.wait) for r in table.to_dict('records')]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i,case in enumerate(pool.map(job,tasks)):
            if i%25==0 or i==len(tasks)-1:print(f'Synthesized {i+1}/{len(tasks)} {case}',flush=True)


if __name__=='__main__':main()
