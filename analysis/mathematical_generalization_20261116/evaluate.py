from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import numpy as np
import pandas as pd

STUDY = Path(__file__).resolve().parents[2]
SOURCE = STUDY / 'data/input_response_20261112'
OUT = STUDY / 'data/mathematical_generalization_20261116'
SOILS = {'sand': (13.0, 2.6, 8e-5), 'loam': (3.6, 1.6, 1e-5)}

def conductivity(h, material):
    alpha, n, ks = SOILS[material]
    m = 1 - 1/n
    x = alpha*np.maximum(-np.asarray(h), 0)
    se = (1+x**n)**(-m)
    return 3.6e6*ks*np.sqrt(se)*(1-(1-se**(1/m))**m)**2

def elasticity(h, material):
    alpha, n, _ = SOILS[material]
    m = 1 - 1/n
    x = (alpha*abs(h))**n
    a = (x/(1+x))**m
    return (n-1)*(.5*x+2*a/(1-a))/(1+x)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = ['input_response_by_configuration.csv', 'hydraulic_histories.csv', 'matched_histories.csv',
             'selected_interval_fluxes.csv', 'temporary_storage.csv', 'selected_signal_contributions.csv',
             'layer_pore_storage.csv', 'moment_responses.csv']
    hashes = {name: hashlib.sha256((SOURCE/name).read_bytes()).hexdigest() for name in files}
    table = pd.read_csv(SOURCE/files[0])
    a = 14.4
    table['eta_0_to_14'] = table.first_increment_mm/a
    table['eta_14_to_28'] = table.second_increment_mm/a
    table['increment_fraction_change'] = table.excess_second_increment_mm/a
    table['curvature_per_mm'] = table.excess_second_increment_mm/a**2
    table['positive_excess_above_0p03_mm'] = table.excess_second_increment_mm > .03
    table.to_csv(OUT/'input_fraction_by_configuration.csv', index=False)
    means = table.groupby('initial_head_m')[['eta_0_to_14', 'eta_14_to_28', 'increment_fraction_change', 'curvature_per_mm']].mean()
    means.to_csv(OUT/'input_fraction_means.csv')

    h = pd.read_csv(SOURCE/'hydraulic_histories.csv')
    matched = pd.read_csv(SOURCE/'matched_histories.csv')
    last = matched[matched.time_h == 72].copy()
    last['delta_input_mm'] = last.input_high_mm-last.input_low_mm
    last['partition_residual_mm'] = last.additional_upper_mm+last.additional_lower_mm+last.additional_drainage_mm-last.delta_input_mm
    last.to_csv(OUT/'matched_final_partition.csv', index=False)

    peak_rows = []
    for order in [0, 1]:
        for amount in [14.4, 28.8]:
            d = matched[(matched.order == order) & (matched.interface_m == [.25, .75][order]) &
                        (matched.initial_head_m == -.5) & (matched.input_high_mm == amount) &
                        (matched.input_low_mm == 0)].sort_values('time_h').reset_index(drop=True)
            k = int(d.additional_lower_mm.idxmax())
            assert 0 < k < len(d)-1
            p = d.iloc[k]
            changes = {}
            for label, i, j in [('before', k-1, k), ('after', k, k+1)]:
                dt = d.time_h.iloc[j]-d.time_h.iloc[i]
                qm = (d.additional_transfer_mm.iloc[j]-d.additional_transfer_mm.iloc[i])/dt
                qb = (d.additional_drainage_mm.iloc[j]-d.additional_drainage_mm.iloc[i])/dt
                storage_rate = (d.additional_lower_mm.iloc[j]-d.additional_lower_mm.iloc[i])/dt
                changes.update({f'{label}_additional_interface_mm_h': qm, f'{label}_additional_basal_mm_h': qb,
                    f'{label}_flux_difference_mm_h': qm-qb, f'{label}_storage_rate_mm_h': storage_rate})
            end = d.iloc[-1]
            peak_rows.append(dict(order=order, input_mm=amount, peak_time_h=p.time_h,
                maximum_additional_lower_mm=p.additional_lower_mm,
                post_peak_additional_transfer_mm=end.additional_transfer_mm-p.additional_transfer_mm,
                post_peak_additional_drainage_mm=end.additional_drainage_mm-p.additional_drainage_mm,
                post_peak_storage_decrease_mm=p.additional_lower_mm-end.additional_lower_mm, **changes))
    peaks = pd.DataFrame(peak_rows); peaks.to_csv(OUT/'storage_peak_balance.csv', index=False)

    constitutive = []
    for material in SOILS:
        for head in [-1.5, -.5, -.15, -.19]:
            eps = 1e-5
            derivative = -(np.log(conductivity(head*np.exp(eps), material))-np.log(conductivity(head*np.exp(-eps), material)))/(2*eps)
            chi = elasticity(head, material)
            assert np.isclose(chi, derivative, rtol=1e-5, atol=1e-6)
            constitutive.append(dict(material=material, head_m=head,
                conductivity_mm_h=float(conductivity(head, material)), suction_elasticity=chi,
                numerical_elasticity=float(derivative)))
    pd.DataFrame(constitutive).to_csv(OUT/'conductivity_sensitivity.csv', index=False)
    selected = h[h.half_hour_output].copy()
    initial = selected[(selected.time_h == -2) & (selected.infiltration_mm == 0)].copy()
    initial['initial_storage_mm'] = initial.upper_mm+initial.lower_mm
    initial['saturation_deficit_mm'] = 430-initial.initial_storage_mm
    initial['input_14_over_deficit'] = 14.4/initial.saturation_deficit_mm
    initial['input_28_over_deficit'] = 28.8/initial.saturation_deficit_mm
    initial[['order','interface_m','initial_storage_mm','saturation_deficit_mm',
             'input_14_over_deficit','input_28_over_deficit']].to_csv(OUT/'selected_storage_scaling.csv',index=False)
    selected['K_from_printed_head_mm_h'] = [float(conductivity(row.base_head_m, 'loam' if row.order == 0 else 'sand')) for row in selected.itertuples()]
    selected['flux_minus_K_mm_h'] = selected.base_flux_mm_h-selected.K_from_printed_head_mm_h
    selected.to_csv(OUT/'selected_basal_states.csv', index=False)

    state_rows = []; integrated = []
    for order, group in selected.groupby('order'):
        t = sorted(group.time_h.unique())
        states = [group[group.infiltration_mm == p].set_index('time_h').loc[t] for p in [0, 14.4, 28.8]]
        head = np.stack([state.base_head_m.values for state in states])
        material = 'loam' if order == 0 else 'sand'
        K = conductivity(head, material)
        dh1, dh2 = np.diff(head, axis=0)
        q1, q2 = np.diff(K, axis=0)
        eps = 1e-6
        slopes = []
        for j, dh in enumerate([dh1, dh2]):
            mid = (head[j]+head[j+1])/2
            limit = (conductivity(mid+eps, material)-conductivity(mid-eps, material))/(2*eps)
            slopes.append(np.divide(K[j+1]-K[j], dh, out=limit.copy(), where=abs(dh)>1e-10))
        mean_dh = (dh1+dh2)/2; mean_slope = (slopes[0]+slopes[1])/2
        conductivity_term = mean_dh*(slopes[1]-slopes[0])
        head_response_term = mean_slope*(dh2-dh1)
        assert np.max(abs(conductivity_term+head_response_term-(q2-q1))) < 1e-10
        for k, time in enumerate(t):
            state_rows.append(dict(order=order,time_h=time,first_head_increment_m=dh1[k],second_head_increment_m=dh2[k],
                first_K_increment_mm_h=q1[k],second_K_increment_mm_h=q2[k],
                conductivity_secant_term_mm_h=conductivity_term[k],head_increment_term_mm_h=head_response_term[k]))
        true_excess = states[2].drainage_mm.iloc[-1]-2*states[1].drainage_mm.iloc[-1]+states[0].drainage_mm.iloc[-1]
        reconstructed = np.trapezoid(q2-q1, t)
        integrated.append(dict(order=order,drainage_excess_mm=true_excess,
            from_printed_head_half_hour_integration_mm=reconstructed,
            conductivity_secant_term_mm=np.trapezoid(conductivity_term,t),
            head_increment_term_mm=np.trapezoid(head_response_term,t),
            quadrature_and_printing_difference_mm=reconstructed-true_excess))
    pd.DataFrame(state_rows).to_csv(OUT/'basal_state_divided_differences.csv', index=False)
    pd.DataFrame(integrated).to_csv(OUT/'basal_state_integrals.csv', index=False)

    signal = pd.read_csv(SOURCE/'selected_signal_contributions.csv')
    pore = pd.read_csv(SOURCE/'layer_pore_storage.csv')
    rows = []
    for row in signal.itertuples():
        if row.time_h not in [6, 24, 48, 72]: continue
        common = pore[(pore.order == row.order) & (pore.interface_m == row.interface_m) &
                      (pore.initial_head_m == row.initial_head_m) & (pore.time_h == row.time_h)]
        rec = dict(order=row.order, time_h=row.time_h, q_ams=row.q_ams)
        water = []
        for layer in ['upper', 'lower']:
            by = common[common.layer == layer].set_index('infiltration_mm')
            delta = by.loc[28.8, 'capillary_storage_mm']-by.loc[14.4, 'capillary_storage_mm']
            projection = getattr(row, layer+'_projection_nV')
            rec[layer+'_additional_water_mm'] = delta
            rec[layer+'_projection_nV'] = projection
            rec[layer+'_projected_nV_per_mm'] = projection/delta if abs(delta)>1e-6 else np.nan
            water.append(delta)
        rec['upper_water_fraction'] = water[0]/sum(water)
        rec['upper_voltage_projection_fraction'] = row.upper_projection_nV/row.complex_difference_magnitude_nV
        rec['upper_to_lower_response_per_water'] = rec['upper_projected_nV_per_mm']/rec['lower_projected_nV_per_mm']
        rows.append(rec)
    nmr = pd.DataFrame(rows); nmr.to_csv(OUT/'signal_water_weights.csv', index=False)
    report = dict(time=datetime.now(ZoneInfo('America/Los_Angeles')).isoformat(),
        source_sha256=hashes, reference_configurations=len(table),
        final_partition_max_residual_mm=float(last.partition_residual_mm.abs().max()),
        mean_dimensionless_responses=means.reset_index().to_dict('records'),
        intermediate_configurations_with_excess_above_0p03_mm=int(table[table.initial_head_m == -.5].positive_excess_above_0p03_mm.sum()),
        storage_peaks=peaks.to_dict('records'), basal_state_integrals=integrated,
        signal_at_24h=nmr[nmr.time_h == 24].to_dict('records'),
        status='Exact identities and conditional derivatives assessed using existing results. No universal response curve fitted. Manuscript unchanged.')
    (OUT/'numerical_findings.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'},indent=2))

if __name__ == '__main__': main()
