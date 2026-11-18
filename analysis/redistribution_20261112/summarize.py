from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

STUDY=Path(__file__).resolve().parents[2]
DATA=STUDY/'data/redistribution_20261112'
WORK=STUDY/'work/redistribution_20261112'
sys.path.insert(0,str(Path(__file__).parent))
from extract import read_tlevel


def main():
    numerical=[]
    for order in (0,1):
        case=f'mechanism_o{order}_d1_h1_r144'
        values={}
        for tag in ('standard','pilot5','mesh800'):
            file=DATA/'hydraulics'/tag/(case+'.npz')
            if not file.exists(): continue
            h=dict(np.load(file)); tl,_=read_tlevel(WORK/'hydrus'/tag/case/'T_LEVEL.OUT',h)
            query=np.array([93600.,266400.]);ix=[np.argmin(abs(h['time_s']-t)) for t in query]
            q=-1000*np.interp(query,tl[:,0],tl[:,10])
            upper=h['z_m']<=.5;lower=h['z_m']>=.5
            changes=[np.diff(np.trapezoid(h['theta_node'][ix][:,mask],h['z_m'][mask],axis=1)*1000)[0] for mask in (upper,lower)]
            values[tag]=dict(case_id=case,tag=tag,basal_drainage_mm=q[1]-q[0],upper_change_mm=changes[0],lower_change_mm=changes[1])
        for tag,v in values.items():
            reference = values['pilot5'] if tag == 'mesh800' else values['standard']
            v['comparison_tag'] = 'pilot5' if tag == 'mesh800' else 'standard'
            for key in ('basal_drainage_mm','upper_change_mm','lower_change_mm'):
                v['difference_'+key]=v[key]-reference[key]
            numerical.append(v)
    pd.DataFrame(numerical).to_csv(DATA/'numerical_comparisons.csv',index=False)
    v={}
    verify=json.loads((DATA/'verification.json').read_text())
    v['BalancePrecision']=format(verify['maximum_balance_residual_mm'],'.4f')
    v['LateBalancePrecision']=format(verify['maximum_late_flux_storage_difference_mm'],'.4f')
    v['InputBalancePrecision']=format(verify['maximum_input_contrast_balance_residual_mm'],'.4f')
    v['ReversedInventoryDifference']=format(verify['maximum_reversed_initial_storage_difference_mm'],'.3f')
    v['MaximumInputDifference']=format(verify['maximum_input_difference_mm'],'.4f')
    num=pd.DataFrame(numerical)
    for tag,label in [('pilot5','Step'),('mesh800','Mesh')]:
        selected=num[num.tag==tag]
        if len(selected):
            v[label+'DrainageDifference']=f'{selected.difference_basal_drainage_mm.abs().max():.4f}'
            v[label+'LayerDifference']=f'{selected[["difference_upper_change_mm","difference_lower_change_mm"]].abs().to_numpy().max():.4f}'
    v['CapillaryPrecision']=f'{json.loads((DATA/"pore_verification.json").read_text())["max_capillary_conservation_error"]:.2e}'
    (DATA/'redistribution_values.tex').write_text(''.join('\\newcommand{\\'+k+'}{'+val+'}\n' for k,val in v.items()))
    (DATA/'manuscript_values.json').write_text(json.dumps(v,indent=2)+'\n')
    print(json.dumps(v,indent=2))


if __name__=='__main__': main()
