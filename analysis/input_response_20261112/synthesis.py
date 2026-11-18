from pathlib import Path
import json
import numpy as np
import pandas as pd
STUDY=Path(__file__).resolve().parents[2];DATA=STUDY/'data/input_response_20261112'
def main():
    h=pd.read_csv(DATA/'hydraulic_histories.csv');diff=pd.read_csv(DATA/'matched_histories.csv')
    chosen=pd.read_csv(DATA/'selected_configurations.csv')[['order','interface_m','initial_head_m']]
    time=h.merge(chosen);selected=diff.merge(chosen)
    intervals=[]
    for key,g in time.groupby(['case_id','order','infiltration_mm']):
        g=g.sort_values('time_h');dt=np.diff(g.time_h)
        for j in range(len(dt)):
            intervals.append(dict(case_id=key[0],order=key[1],infiltration_mm=key[2],
                time_h=float((g.time_h.iloc[j]+g.time_h.iloc[j+1])/2),
                transfer_flux_mm_h=float(np.diff(g.transfer_mm)[j]/dt[j]),
                basal_flux_mm_h=float(np.diff(g.drainage_mm)[j]/dt[j]),
                lower_storage_rate_mm_h=float(np.diff(g.lower_mm)[j]/dt[j])))
    pd.DataFrame(intervals).to_csv(DATA/'selected_interval_fluxes.csv',index=False)
    old=pd.read_csv(STUDY/'data/redistribution_20261112/hydraulic_cases.csv').set_index('case_id')
    comparison=[]
    for r in time[time.time_h==72].to_dict('records'):
        o=old.loc[r['case_id']]
        comparison.append(dict(case_id=r['case_id'],drainage_difference_mm=r['drainage_mm']-o.base_drainage_start_72h_mm,
            storage_difference_mm=r['upper_mm']+r['lower_mm']-o.storage_72h_mm))
    pd.DataFrame(comparison).to_csv(DATA/'denser_output_comparison.csv',index=False)
    peak=pd.read_csv(DATA/'temporary_storage.csv').merge(chosen)
    moment=pd.read_csv(DATA/'signal_input_contrasts.csv').merge(chosen)
    moment=moment[(moment.input_high_mm==28.8)&(moment.input_low_mm==14.4)&np.isclose(moment.q_ams,21.24462820428606)].copy()
    moment['lower_projection_fraction']=moment.lower_projection_nV/moment.complex_difference_magnitude_nV
    moment.to_csv(DATA/'selected_signal_contributions.csv',index=False)
    layer=pd.read_csv(DATA/'layer_pore_storage.csv').merge(chosen)
    layer_dif=[]
    for key,g in layer.groupby(['order','time_h','layer']):
        a=g.set_index('infiltration_mm')
        layer_dif.append(dict(order=key[0],time_h=key[1],layer=key[2],
            additional_water_28_minus_14_mm=a.loc[28.8,'capillary_storage_mm']-a.loc[14.4,'capillary_storage_mm'],
            T2g_14_s=a.loc[14.4,'geometric_T2_s'],T2g_28_s=a.loc[28.8,'geometric_T2_s']))
    pd.DataFrame(layer_dif).to_csv(DATA/'selected_pore_changes.csv',index=False)
    report=dict(selection=json.loads((DATA/'hydraulic_findings.json').read_text()),
        temporary_storage=peak.to_dict('records'),layer_pore_changes=layer_dif,
        selected_signal_contributions=moment.to_dict('records'),
        dense_output_max_drainage_difference_mm=max(abs(r['drainage_difference_mm']) for r in comparison),
        dense_output_max_storage_difference_mm=max(abs(r['storage_difference_mm']) for r in comparison),
        selected_peaks_interior=bool(not peak.maximum_at_last_time.any()),
        interpretation='Larger inputs over the same duration increase both amount and delivery rate. All additional quantities compare matched total responses, not water-source fractions.')
    (DATA/'findings.json').write_text(json.dumps(report,indent=2,default=lambda x:x.item())+'\n')
    c=pd.read_csv(DATA/'input_response_by_configuration.csv')
    lines=['\\begin{table}[htbp]\\centering\\small',
        '\\caption{Drainage increments for equal 14.4 mm increases in two-hour input, from input start ($-2$ h) to 72 h after input ends. Each row is one reference configuration.}',
        '\\label{tab:individual_inputs}',
        '\\begin{tabular}{llrrr}\\toprule',
        ' & & & \\multicolumn{2}{c}{Drainage increment (mm)}\\\\\nLayer order & Initial head (m) & Interface (m) & $0$ to $14.4$ mm & $14.4$ to $28.8$ mm\\\\\\midrule']
    for r in c.sort_values(['initial_head_m','order','interface_m']).itertuples():
        lines.append(f'{["Sand/loam","Loam/sand"][r.order]} & {r.initial_head_m:g} & {r.interface_m:.2f} & {r.first_increment_mm:.3f} & {r.second_increment_mm:.3f}\\\\')
    lines+=['\\bottomrule\\end{tabular}\\end{table}']
    (DATA/'table_individual_inputs.tex').write_text('\n'.join(lines)+'\n')
    print('Dense output maximum drainage difference',report['dense_output_max_drainage_difference_mm'])
    print('Dense output maximum storage difference',report['dense_output_max_storage_difference_mm'])
    print('All selected additional-storage peaks occur before 72 h',report['selected_peaks_interior'])
if __name__=='__main__':main()
