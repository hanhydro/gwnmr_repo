from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from pore_integration import cumulative_lookup
STUDY=Path(__file__).resolve().parents[2];DATA=STUDY/'data/input_response_20261112'
sys.path.insert(0,str(STUDY/'analysis'))
from physics import T2_GRID,TIMES
def main():
    tests=[];se=np.unique(np.r_[np.linspace(.001,1,501),.025,.1,.25,.5,.75,.9,.99])
    for material,alpha,n,rho in [('sand',13.,2.6,6e-6),('loam',3.6,1.6,8e-6)]:
        base_x,base,_=cumulative_lookup(alpha,n,rho)
        base=interp1d(base_x,base,axis=0)(se)
        for points,quad,name in [(1025,32,'quadrature'),(2049,16,'saturation intervals')]:
            x,table,_=cumulative_lookup(alpha,n,rho,points,quad)
            fine=interp1d(x,table,axis=0)(se)
            decay=np.exp(-TIMES[:,None]/T2_GRID)
            signal_error=float(abs((fine-base)@decay.T/se[:,None]).max())
            tests.append(dict(material=material,comparison=name,points=points,quadrature=quad,
                max_normalized_decay_difference=signal_error,
                max_component_water_difference=float(abs(fine-base).max()),
                max_water_conservation_difference=float(abs(fine.sum(axis=1)-se).max())))
    pd.DataFrame(tests).to_csv(DATA/'pore_integration_precision.csv',index=False)
    old=DATA/'previous_pore_moments.csv.gz'
    old=pd.read_csv(old);new=pd.read_csv(DATA/'moment_responses.csv')
    key=['case_id','time_h','q_ams'];a=old.set_index(key).sort_index();b=new.set_index(key).sort_index()
    assert a.index.equals(b.index)
    report=dict(nested_component_distribution=True,settings='1024 fixed pore-volume intervals, each integrated with 16 Gaussian nodes.',
        precision=tests,maximum_5ms_voltage_change_nV=float(abs(a.total_magnitude_nV-b.total_magnitude_nV).max()),
        maximum_profile_mean_change_percent=float(abs(100*(b.profile_T2g_s/a.profile_T2g_s-1)).max()))
    (DATA/'pore_integration_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
