from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd

STUDY=Path(__file__).resolve().parents[2]
DATA=STUDY/'data/input_response_20261112'
OLD=STUDY/'data/redistribution_20261112'
WORK=STUDY/'work/input_response_20261112'
sys.path.insert(0,str(STUDY/'analysis/redistribution_20261112'))
from extract import read_tlevel

def main():
    DATA.mkdir(exist_ok=True,parents=True)
    manifest=pd.read_csv(OLD/'manifest.csv');manifest=manifest[manifest.split=='mechanism']
    cases=pd.read_csv(OLD/'hydraulic_cases.csv')
    case_rows=[]
    for key,g in cases[cases.population=='reference'].groupby(['order','interface_m','initial_head_m']):
        a=g.set_index('infiltration_mm');d=a.base_drainage_start_72h_mm
        case_rows.append(dict(order=key[0],interface_m=key[1],initial_head_m=key[2],
            drainage_zero_mm=d[0],drainage_14_mm=d[14.4],drainage_28_mm=d[28.8],
            first_increment_mm=d[14.4]-d[0],second_increment_mm=d[28.8]-d[14.4],
            excess_second_increment_mm=d[28.8]-2*d[14.4]+d[0]))
    case_table=pd.DataFrame(case_rows)
    case_table.to_csv(DATA/'input_response_by_configuration.csv',index=False)
    chosen=case_table[case_table.initial_head_m==-.5].sort_values('excess_second_increment_mm').groupby('order').tail(1).sort_values('order')
    chosen.to_csv(DATA/'selected_configurations.csv',index=False)
    selection=manifest.merge(chosen[['order','interface_m','initial_head_m']],on=['order','interface_m','initial_head_m'])
    selection.to_csv(DATA/'selected_manifest.csv',index=False)
    histories=[];profiles=[];hashes={};errors=[]
    for r in manifest.to_dict('records'):
        cid=r['case_id'];src=OLD/'hydraulics/standard'/(cid+'.npz')
        fine=DATA/'hydraulics/half_hour'/(cid+'.npz')
        using_fine=fine.exists() and json.loads(fine.with_suffix('.json').read_text()).get('accepted')
        path=fine if using_fine else src
        h=dict(np.load(path));z=h['z_m'];mask=z<=r['interface_m']
        native=(WORK if using_fine else STUDY/'work/redistribution_20261112')/'hydrus'/('half_hour' if using_fine else 'standard')/cid
        tl,err=read_tlevel(native/'T_LEVEL.OUT',h)
        times=(h['time_s']-7200)/3600
        upper=1000*np.trapezoid(h['theta_node'][:,mask],z[mask],axis=1)
        lower=1000*np.trapezoid(h['theta_node'][:,~(z<r['interface_m'])],z[~(z<r['interface_m'])],axis=1)
        rain=-1000*np.interp(h['time_s'],tl[:,0],tl[:,8]);rain[0]=0
        drain=-1000*np.interp(h['time_s'],tl[:,0],tl[:,10]);drain[0]=0
        transfer=rain+upper[0]-upper
        basal=-3.6e6*np.interp(h['time_s'],tl[:,0],tl[:,5])
        common={k:r[k] for k in ['case_id','order','interface_m','initial_head_m','infiltration_mm']}
        for i,t in enumerate(times):
            histories.append(dict(**common,time_h=t,upper_mm=upper[i],lower_mm=lower[i],
                input_mm=rain[i],drainage_mm=drain[i],transfer_mm=transfer[i],
                base_head_m=h['head_node'][i,-1],base_flux_mm_h=basal[i],
                half_hour_output=using_fine))
            if t in [0,6,24,48,72]:
                for j in range(len(z)):
                    tr,ts,alpha,n,ks=h['parameters'][h['material_node'][j]];m=1-1/n
                    head=h['head_node'][i,j];se=(1+(alpha*max(-head,0))**n)**(-m)
                    conductivity=ks*np.sqrt(se)*(1-(1-se**(1/m))**m)**2*3.6e6
                    profiles.append(dict(**common,time_h=t,depth_m=z[j],theta=h['theta_node'][i,j],
                        head_m=head,conductivity_mm_h=conductivity))
        residual=upper+lower-upper[0]-lower[0]-rain+drain
        errors.append(dict(case_id=cid,max_balance_residual_mm=float(abs(residual).max()),
            half_hour_output=using_fine))
        hashes[str(path.relative_to(STUDY))]=hashlib.sha256(path.read_bytes()).hexdigest()
    history=pd.DataFrame(histories);history.to_csv(DATA/'hydraulic_histories.csv',index=False)
    pd.DataFrame(profiles).to_csv(DATA/'hydraulic_profiles.csv',index=False)
    differences=[]
    for keys,g in history.groupby(['order','interface_m','initial_head_m']):
        for a,b in [(14.4,0),(28.8,0),(28.8,14.4)]:
            x=g[g.infiltration_mm==a].set_index('time_h');y=g[g.infiltration_mm==b].set_index('time_h')
            assert np.array_equal(x.index,y.index)
            for t in x.index:
                rec=dict(order=keys[0],interface_m=keys[1],initial_head_m=keys[2],
                    input_high_mm=a,input_low_mm=b,time_h=t)
                for k in ['upper_mm','lower_mm','drainage_mm','transfer_mm','base_head_m','base_flux_mm_h']:
                    rec['additional_'+k]=x.loc[t,k]-y.loc[t,k]
                differences.append(rec)
    dif=pd.DataFrame(differences);dif.to_csv(DATA/'matched_histories.csv',index=False)
    peaks=[]
    for keys,g in dif.groupby(['order','interface_m','initial_head_m','input_high_mm','input_low_mm']):
        g=g.sort_values('time_h');peak=g.loc[g.additional_lower_mm.idxmax()];end=g.iloc[-1]
        peaks.append(dict(order=keys[0],interface_m=keys[1],initial_head_m=keys[2],input_high_mm=keys[3],input_low_mm=keys[4],
            maximum_additional_lower_mm=peak.additional_lower_mm,peak_time_h=peak.time_h,
            maximum_at_last_time=bool(peak.time_h==72),lower_72_mm=end.additional_lower_mm,
            drainage_at_peak_mm=peak.additional_drainage_mm,
            drainage_after_peak_mm=end.additional_drainage_mm-peak.additional_drainage_mm))
    pd.DataFrame(peaks).to_csv(DATA/'temporary_storage.csv',index=False)
    mean=case_table.groupby('initial_head_m')[['first_increment_mm','second_increment_mm','excess_second_increment_mm']].mean()
    mean['first_fraction']=mean.first_increment_mm/14.4;mean['second_fraction']=mean.second_increment_mm/14.4
    mean.to_csv(DATA/'input_response_means.csv')
    pd.DataFrame(errors).to_csv(DATA/'water_balance_checks.csv',index=False)
    report=dict(selected_rule='Largest second-minus-first drainage increment within each layer order at initial head -0.5 m.',
        selected=chosen.to_dict('records'),share_of_intermediate_excess=float(chosen.excess_second_increment_mm.sum()/case_table[case_table.initial_head_m==-.5].excess_second_increment_mm.sum()),
        maximum_balance_residual_mm=max(r['max_balance_residual_mm'] for r in errors),
        means=mean.reset_index().to_dict('records'),source_sha256=hashes)
    (DATA/'hydraulic_findings.json').write_text(json.dumps(report,indent=2)+'\n')
    print(mean.to_string());print(chosen.to_string(index=False));print('Selected excess share',report['share_of_intermediate_excess'])

if __name__=='__main__':main()
