from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
STUDY=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(STUDY/'analysis/redistribution_20261112'))
from signal_model import convert,operator
from physics import T2_GRID,TIMES
sys.path.insert(0,str(Path(__file__).resolve().parent))
from pore_integration import cumulative_lookup
import signals
signals.lookup=cumulative_lookup
DATA=STUDY/'data/input_response_20261112';OLD=STUDY/'data/redistribution_20261112'
HOURS=[0,6,24,48,72]

def main():
    threadpool_limits(1)
    manifest=pd.read_csv(OLD/'manifest.csv');manifest=manifest[manifest.split=='mechanism']
    selected=pd.read_csv(DATA/'selected_manifest.csv').case_id.tolist()
    a,bg,q=operator();iq=int(np.argmin(abs(q-21.24462820428606)))
    stores=[];spectra=[];profiles=[];voltages=[];fids=[];decompositions=[]
    max_error=0.;max_component_error=0.;data_cache={}
    (DATA/'signals').mkdir(exist_ok=True)
    for n,row in enumerate(manifest.to_dict('records')):
        cid=row['case_id'];path=DATA/'hydraulics/half_hour'/(cid+'.npz')
        if not path.exists():path=OLD/'hydraulics/standard'/(cid+'.npz')
        h=dict(np.load(path));ix=[np.argmin(abs(h['time_s']-(7200+t*3600))) for t in HOURS]
        assert np.max(abs(h['time_s'][ix]-(7200+np.array(HOURS)*3600)))<.1
        for key in ('theta_node','head_node','time_s','storage_m'):h[key]=h[key][ix]
        d,total,cap,error,params=convert(h,row,'standard');max_error=max(max_error,error)
        split=int(round(row['interface_m']/.025));v=[];layer_mm=[];layer_log=[]
        for name,section in [('upper',slice(None,split)),('lower',slice(split,None))]:
            dist=d[:,section,:].sum(axis=1)*25
            amount=dist.sum(axis=1);lg=(dist@np.log(T2_GRID))/amount
            layer_mm.append(amount);layer_log.append(lg)
            v.append(np.einsum('wzk,qtzk->wqt',d[:,section,:],a[:,:,section,:],optimize=True))
            for it,t in enumerate(HOURS):
                base=dict(case_id=cid,order=row['order'],interface_m=row['interface_m'],initial_head_m=row['initial_head_m'],
                    infiltration_mm=row['infiltration_mm'],time_h=t,layer=name)
                stores.append(dict(**base,capillary_storage_mm=amount[it],geometric_T2_s=float(np.exp(lg[it])),
                    hydraulic_capillary_mm=float(cap[it,section].sum()*25)))
                for k in range(len(T2_GRID)):
                    spectra.append(dict(**base,T2_s=T2_GRID[k],capillary_water_mm=dist[it,k]))
        whole=v[0]+v[1]+bg
        direct=np.einsum('wzk,qtzk->wqt',d,a,optimize=True)+bg
        max_component_error=max(max_component_error,float(abs(whole-direct).max()))
        mm=np.array(layer_mm).T;lg=np.array(layer_log).T;p=mm/mm.sum(axis=1,keepdims=True)
        g=np.exp((p*lg).sum(axis=1))
        data_cache[cid]=(p,lg,v,whole)
        np.savez_compressed(DATA/'signals'/(cid+'.npz'),distribution=d,upper_voltage=v[0],lower_voltage=v[1],
            total_voltage=whole,background_voltage=bg,q_ams=q,recording_time_s=TIMES,time_h=HOURS)
        for it,t in enumerate(HOURS):
            base=dict(case_id=cid,order=row['order'],interface_m=row['interface_m'],initial_head_m=row['initial_head_m'],
                    infiltration_mm=row['infiltration_mm'],time_h=t)
            for j in range(40):
                gm=np.exp(d[it,j]@np.log(T2_GRID)/d[it,j].sum())
                profiles.append(dict(**base,depth_m=(j+.5)*.025,theta_capillary=cap[it,j],geometric_T2_s=gm))
            for j,moment in enumerate(q):
                voltages.append(dict(**base,q_ams=moment,profile_T2g_s=g[it],
                    upper_real_nV=v[0][it,j,0].real,upper_imag_nV=v[0][it,j,0].imag,
                    lower_real_nV=v[1][it,j,0].real,lower_imag_nV=v[1][it,j,0].imag,
                    total_real_nV=whole[it,j,0].real,total_imag_nV=whole[it,j,0].imag,
                    total_magnitude_nV=abs(whole[it,j,0])))
            if cid in selected:
                for k,tau in enumerate(TIMES):
                    fids.append(dict(**base,q_ams=q[iq],recording_time_s=tau,
                        total_real_nV=whole[it,iq,k].real,total_imag_nV=whole[it,iq,k].imag))
        for lo,hi in [(2,4)]:
            loc=((p[lo]+p[hi])/2)*(lg[hi]-lg[lo]);wt=((lg[lo]+lg[hi])/2)*(p[hi]-p[lo])
            assert abs((loc+wt).sum()-np.log(g[hi]/g[lo]))<1e-12
            decompositions.append(dict(case_id=cid,comparison='72 h minus 24 h',
                within_upper_log=loc[0],within_lower_log=loc[1],weight_upper_log=wt[0],weight_lower_log=wt[1],
                total_log_change=(loc+wt).sum()))
        if n%9==0:print('Pore and layer-signal cases',n+1,'/',len(manifest),flush=True)
    pd.DataFrame(stores).to_csv(DATA/'layer_pore_storage.csv',index=False)
    pd.DataFrame(spectra).to_csv(DATA/'pore_water_distributions.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    pd.DataFrame(profiles).to_csv(DATA/'pore_profiles.csv',index=False)
    pd.DataFrame(voltages).to_csv(DATA/'moment_responses.csv',index=False)
    pd.DataFrame(fids).to_csv(DATA/'selected_fids.csv',index=False)
    pd.DataFrame(decompositions).to_csv(DATA/'mean_decomposition.csv',index=False)
    contrasts=[];contributions=[]
    for keys,group in manifest.groupby(['order','interface_m','initial_head_m']):
        ids=group.set_index('infiltration_mm').case_id
        for high,low in [(14.4,0),(28.8,0),(28.8,14.4)]:
            p1,l1,v1,e1=data_cache[ids[high]];p0,l0,v0,e0=data_cache[ids[low]]
            for it,t in enumerate(HOURS):
                for j,moment in enumerate(q):
                    du=v1[0][it,j,0]-v0[0][it,j,0];dl=v1[1][it,j,0]-v0[1][it,j,0]
                    dt=e1[it,j,0]-e0[it,j,0]
                    assert abs(du+dl-dt)<1e-12
                    axis=np.conj(dt)/abs(dt) if abs(dt)>1e-14 else 0j
                    contrasts.append(dict(order=keys[0],interface_m=keys[1],initial_head_m=keys[2],
                        input_high_mm=high,input_low_mm=low,time_h=t,q_ams=moment,
                        magnitude_difference_nV=abs(e1[it,j,0])-abs(e0[it,j,0]),
                        complex_difference_magnitude_nV=abs(dt),upper_projection_nV=float((du*axis).real),
                        lower_projection_nV=float((dl*axis).real),
                        upper_delta_real_nV=du.real,upper_delta_imag_nV=du.imag,
                        lower_delta_real_nV=dl.real,lower_delta_imag_nV=dl.imag))
                loc=(p1[it]+p0[it])/2*(l1[it]-l0[it]);wt=(l1[it]+l0[it])/2*(p1[it]-p0[it])
                contributions.append(dict(order=keys[0],interface_m=keys[1],initial_head_m=keys[2],
                    input_high_mm=high,input_low_mm=low,time_h=t,within_upper_log=loc[0],within_lower_log=loc[1],
                    weight_upper_log=wt[0],weight_lower_log=wt[1],total_log_change=(loc+wt).sum()))
    pd.DataFrame(contrasts).to_csv(DATA/'signal_input_contrasts.csv',index=False)
    pd.DataFrame(contributions).to_csv(DATA/'mean_input_decomposition.csv',index=False)
    report=dict(cases=len(manifest),times_h=HOURS,pulse_moments=len(q),max_capillary_error_m3m3=max_error,
        max_complex_component_sum_error_nV=max_component_error,fixed_pulse_moment_ams=float(q[iq]),
        signal_definition='Upper-layer plus lower-layer complex voltage plus the fixed deeper-water contribution.',
        mean_definition='Capillary-component volumes pooled with 0.025 m interval thickness. Both local and layer means use the same components.',
        pore_integration='Cumulative integration across 1024 fixed pore-volume intervals with 16 Gauss points per interval. Lookup interpolation retains nested occupied-pore distributions.',
        component_projection='Real projection of each complex layer change onto the direction of their total change. The sum equals the norm of the total complex change, not a difference of full-signal magnitudes.')
    (DATA/'pore_verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
