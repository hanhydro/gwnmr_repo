from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy.stats import qmc

STUDY = Path(__file__).resolve().parents[2]
DATA = STUDY/'data/physical_recovery_20260929'
HEADS = [-1.5, -.5, -.15]
MOMENTS = dict(full=list(range(24)),
               broad=[0,4,9,14,19,23],
               shallow=[0,2,5,7,10,12],
               middle=[6,9,12,15,18,21],
               deep=[12,14,16,18,20,23])
SPLITS = dict(library=128, development=16, confirmation=24)
PROPERTY_NAMES = [f'{p}_{s}_scale' for s in ('sand','loam') for p in ('alpha','ks','rho')]


def run():
    DATA.mkdir(exist_ok=True, parents=True)
    records=[]
    for order in (0,1):
        cells=np.random.default_rng(202609291+order).permutation(np.arange(60,340))
        offset=0
        for j,(split,count) in enumerate(SPLITS.items()):
            coordinates=qmc.LatinHypercube(6,seed=202609292+100*j+order).random(count)
            for k in range(count):
                gid=f'{split}_o{order}_{k:03d}'
                props=dict(zip(PROPERTY_NAMES,.8+.4*coordinates[k]))
                depth=(float(cells[offset+k])+.5)/400
                for state,head in enumerate(HEADS):
                    records.append(dict(case_id=f'{gid}_h{state}',geometry_id=gid,split=split,
                                        order=order,interface_m=depth,state=state,
                                        initial_head_m=head,infiltration_mm=14.4,**props))
            offset+=count
    for order in (0,1):
        for d,depth in enumerate((.25,.5,.75)):
            for state,head in enumerate(HEADS):
                for rain in (0.,14.4,28.8):
                    gid=f'mechanism_o{order}_d{d}'
                    records.append(dict(case_id=f'{gid}_h{state}_r{int(rain*10):03d}',
                                        geometry_id=gid,split='mechanism',order=order,
                                        interface_m=depth,state=state,initial_head_m=head,
                                        infiltration_mm=rain,**dict.fromkeys(PROPERTY_NAMES,1.)))
    table=pd.DataFrame(records)
    path=DATA/'manifest.csv'
    if path.exists():
        assert path.read_text()==table.to_csv(index=False), 'Refuse to replace a changed design'
    else:
        table.to_csv(path,index=False)
    specification=dict(created_utc=datetime.now(timezone.utc).isoformat(),
       manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
       sizes=table.groupby('split').size().to_dict(),
       observation_starts_h=dict(late=[0,6,24],early=[0,3,6]),
       pulse_moment_candidates=MOMENTS,total_excitations=576,
       excitations_per_sounding=192,repetition_interval_s=16,
       duration_per_sounding_s=3072,acquisition_total_h=2.56,
       per_scan_component_sigma_nV=float(.05*np.sqrt(72)),
       field_spread_rad_s=5,development_noise_repetitions=12,
       confirmation_noise_repetitions=20,
       inverse_strength_candidates=[1.,3.,10.],
       likelihood_temperatures=[1.,2.,4.,8.,16.,32.,64.,128.],
       selection='Common inverse smoothing by development capillary-water error; likelihood temperature separately for each input by development interface RMSE; six moments by joint development interface RMSE under the late schedule. Evaluate selected six-moment early schedule and full late schedule separately.',
       primary_confirmation='full_late',
       secondary_confirmation=['selected_six_late','selected_six_early'],
       primary_targets=['interface depth (m)','upper-meter total water storage at 72 h (mm)'],
       uncertainty='Paired bootstrap over complete confirmation geometries, averaging losses over states and noise first. Report ensemble 90% interval coverage without assuming calibration.',
       initial_state_prior='Three uniform heads equiprobable; true head and true order are not supplied to inference.',
       shared_boundary='Each candidate has one fixed interface, one material order, and one set of material properties for its complete hydraulic trajectory.',
       numerical_acceptance='Complete profiles; finite water content; maximum absolute reported mass-balance error <=0.2 mm. Retry at dtmax=5 s on failure, without changing the sampled soil.',
       pending='Freeze inverse settings, covariance, candidate moments, and likelihood temperatures before running confirmation inference.')
    out=DATA/'design.json'
    if not out.exists():out.write_text(json.dumps(specification,indent=2)+'\n')
    print(table.groupby('split').size().to_string())


if __name__=='__main__':run()
