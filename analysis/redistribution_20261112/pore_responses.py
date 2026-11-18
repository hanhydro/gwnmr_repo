from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

STUDY=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(STUDY/'analysis/physical_recovery_20260929'),str(STUDY/'analysis')]
from signal_model import convert,operator
from physics import TIMES,T2_GRID
OUT=STUDY/'data/redistribution_20261112'
SOURCE=STUDY/'data/redistribution_20261112'


def main():
    threadpool_limits(1)
    rows=pd.read_csv(SOURCE/'manifest.csv')
    rows=rows[(rows.split=='mechanism')&(rows.infiltration_mm==14.4)]
    a,bg,q=operator()
    profiles,signals,summaries,distributions=[],[],[],[]
    max_error=0.
    for row in rows.to_dict('records'):
        h=dict(np.load(SOURCE/'hydraulics/standard'/(row['case_id']+'.npz')))
        ix=[np.argmin(abs(h['time_s']-(7200+t*3600))) for t in (0,24,72)]
        for key in ('theta_node','head_node','time_s','storage_m'):
            h[key]=h[key][ix]
        d,total,cap,error,params=convert(h,row,'standard')
        max_error=max(max_error,error)
        voltage=np.einsum('wzk,qtzk->wqt',d,a,optimize=True)
        gm=np.exp(np.sum(d*np.log(T2_GRID),axis=-1)/d.sum(axis=-1))
        for it,time in enumerate((0,24,72)):
            common=dict(case_id=row['case_id'],order=row['order'],interface_m=row['interface_m'],
                        initial_head_m=row['initial_head_m'],time_h=time)
            for j in range(40):
                profiles.append(dict(**common,depth_m=(j+.5)*.025,theta_total=total[it,j],
                                     theta_capillary=cap[it,j],geometric_T2_s=gm[it,j]))
            for iq,moment in enumerate(q):
                for jt,t in enumerate(TIMES):
                    v=voltage[it,iq,jt];vtotal=v+bg[iq,jt]
                    signals.append(dict(**common,q_ams=moment,recording_time_s=t,upper_real_nV=v.real,
                        upper_imag_nV=v.imag,total_real_nV=vtotal.real,total_imag_nV=vtotal.imag))
            dist=d[it].mean(axis=0)
            for k,t2 in enumerate(T2_GRID):
                distributions.append(dict(**common,T2_s=t2,capillary_volume_fraction=dist[k]))
            summaries.append(dict(**common,storage_mm=total[it].mean()*1000,
                capillary_storage_mm=cap[it].mean()*1000,
                geometric_T2_s=float(np.exp(dist@np.log(T2_GRID)/dist.sum())),
                intrinsic_fraction_at_5ms=float(dist@np.exp(-.005/T2_GRID)/dist.sum())))
    for name,records in [('pore_profiles',profiles),('fid_snapshots',signals),
                         ('relaxation_distributions',distributions),('pore_summary',summaries)]:
        pd.DataFrame(records).to_csv(OUT/(name+'.csv'),index=False)
    f=pd.DataFrame(signals)
    selected=f[np.isclose(f.interface_m,.5)&(f.time_h==24)&np.isclose(f.recording_time_s,TIMES[0])].copy()
    selected['amplitude']=np.hypot(selected.upper_real_nV,selected.upper_imag_nV)
    chosen=selected.groupby('q_ams').amplitude.mean().idxmax()
    record=dict(cases=len(rows),max_capillary_conservation_error= max_error,
        figure_pulse_moment_ams=float(chosen),
        signal_times='Instantaneous physical states at 0, 24 and 72 h after infiltration. These snapshots are not sequential soundings.',
        shallow_signal='The main temporal figure uses the full complex voltage including stationary 1–10 m water. The upper-meter component is supplied separately.',
        pore_model='Circular capillaries derived from the retention relation, reference relaxivities 6 and 8 micrometers per second. Equal-relaxivity material control is separate.')
    (OUT/'pore_verification.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__': main()
