from pathlib import Path
import sys
import json
import hashlib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

STUDY=Path(__file__).resolve().parents[2]
DATA=STUDY/'data/redistribution_20261112'
sys.path[:0]=[str(STUDY/'analysis/physical_recovery_20260929'),str(STUDY/'analysis')]
from signal_model import convert,operator
from physics import T2_GRID


def main():
    threadpool_limits(1)
    manifest=pd.read_csv(DATA/'manifest.csv').set_index('case_id')
    a,bg,q=operator()
    chosen=json.loads((DATA/'pore_verification.json').read_text())['figure_pulse_moment_ams']
    iq=np.argmin(abs(q-chosen))
    rows=[]
    for order in (0,1):
        case=f'mechanism_o{order}_d1_h1_r144'
        row=manifest.loc[case].to_dict();row['case_id']=case
        for tag in ('standard','pilot5','mesh800'):
            file=DATA/'hydraulics'/tag/(case+'.npz')
            h=dict(np.load(file)); ix=[np.argmin(abs(h['time_s']-t)) for t in (93600,266400)]
            for key in ('theta_node','head_node','time_s','storage_m'):h[key]=h[key][ix]
            d,total,cap,error,params=convert(h,row,tag)
            dist=d.mean(axis=1)
            gm=np.exp(dist@np.log(T2_GRID)/dist.sum(axis=-1))
            voltage=np.einsum('wzk,tzk->wt',d,a[iq],optimize=True)+bg[iq]
            mag=np.abs(voltage)
            normalized=mag/mag[:,0,None]
            rows.append(dict(case_id=case,tag=tag,q_ams=q[iq],
                T2g_change_percent=100*(gm[1]/gm[0]-1),
                signal_5ms_change_percent=100*(mag[1,0]/mag[0,0]-1),
                max_normalized_shape_change_pp=float(np.max(abs(normalized[1]-normalized[0]))*100),
                capillary_conservation_error=error,input_sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
    table=pd.DataFrame(rows)
    for idx,row in table.iterrows():
        ref_tag='pilot5' if row.tag=='mesh800' else 'standard'
        ref=table[(table.case_id==row.case_id)&(table.tag==ref_tag)].iloc[0]
        table.loc[idx,'comparison_tag']=ref_tag
        for key in ('T2g_change_percent','signal_5ms_change_percent','max_normalized_shape_change_pp'):
            table.loc[idx,'difference_'+key]=row[key]-ref[key]
    table.to_csv(DATA/'signal_numerical_comparisons.csv',index=False)
    print(table.drop(columns=['input_sha256']).to_string(index=False))


if __name__=='__main__':main()
