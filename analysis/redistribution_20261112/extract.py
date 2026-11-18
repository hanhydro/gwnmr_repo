from pathlib import Path
import hashlib
import json
import re
import zipfile
import argparse
import numpy as np
import pandas as pd

STUDY = Path(__file__).resolve().parents[2]
OUT = STUDY / 'data/redistribution_20261112'
SOURCE = STUDY / 'data/redistribution_20261112'
HOURS = np.array([0., 6., 24., 48., 72.])


def native_bytes(path):
    if path.exists():
        return path.read_bytes()
    with zipfile.ZipFile(OUT/'native_outputs.zip') as archive:
        return archive.read(str(path.relative_to(STUDY)))


def read_tlevel(path, h):
    rows = []
    for line in native_bytes(path).decode().splitlines():
        p = line.split()
        if len(p) < 22:
            continue
        try:
            rows.append([float(x) for x in p[:22]])
        except ValueError:
            pass
    a = np.array(rows)
    segments = np.split(a, np.flatnonzero(np.diff(a[:, 0]) <= 0) + 1)
    full = [s for s in segments if len(s) > 100 and s[0, 0] <= 7200 and s[-1, 0] >= 266400]
    errors = [np.max(abs(h['storage_m'][1:] - np.interp(h['time_s'][1:], s[:, 0], s[:, 16]))) for s in full]
    assert full and min(errors) < .0002, path
    return full[int(np.argmin(errors))], min(errors) * 1000


def node_blocks(path):
    blocks, time, rows = [], None, []
    for line in native_bytes(path).decode().splitlines():
        match = re.search(r'Time:\s*([\d.Ee+\-]+)', line)
        if match:
            if rows:
                blocks.append((time, np.array(rows)))
            time, rows = float(match[1]), []
        elif time is not None:
            p = line.split()
            if len(p) >= 11:
                try:
                    row = [float(x) for x in p[:11]]
                    if row[0].is_integer() and row[0] >= 1:
                        rows.append(row)
                except ValueError:
                    pass
    if rows:
        blocks.append((time, np.array(rows)))
    return blocks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference-only', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    groups = [('reference', 'redistribution_20261112', 'mechanism'),
              ('variable', 'redistribution_20261112', 'confirmation'),
              ('depth', 'redistribution_20261112', 'confirmation_depth')]
    if args.reference_only:
        groups = groups[:1]
    records, histories, checks, hashes, profiles = [], [], [], {}, []
    for population, folder, split in groups:
        source = STUDY / 'data' / folder
        manifest = pd.read_csv(source / 'manifest.csv')
        manifest = manifest[manifest.split == split]
        for row in manifest.to_dict('records'):
            case = row['case_id']
            file = source / 'hydraulics/standard' / (case + '.npz')
            h = dict(np.load(file))
            native = STUDY / 'work' / folder / 'hydrus/standard' / case
            tl, err = read_tlevel(native / 'T_LEVEL.OUT', h)
            hashes[str(file.relative_to(STUDY))] = hashlib.sha256(file.read_bytes()).hexdigest()
            hashes[str((native / 'T_LEVEL.OUT').relative_to(STUDY))] = hashlib.sha256(native_bytes(native / 'T_LEVEL.OUT')).hexdigest()
            query = 7200 + HOURS * 3600
            storage = np.interp(query, h['time_s'], h['storage_m']) * 1000
            flux_total = -1000 * np.interp(query, tl[:, 0], tl[:, 10])
            top = -1000 * np.interp(query, tl[:, 0], tl[:, 8])
            initial = float(h['storage_m'][0] * 1000)
            rec = dict(population=population, **row, initial_storage_mm=initial)
            for hour, s, q in zip(HOURS, storage, flux_total):
                rec[f'storage_{int(hour)}h_mm'] = s
                rec[f'base_drainage_start_{int(hour)}h_mm'] = q
            rec['drainage_24_72_mm'] = storage[2] - storage[-1]
            rec['base_drainage_24_72_mm'] = flux_total[-1] - flux_total[2]
            rec['actual_input_mm'] = top[0]
            rec['initial_sand_fraction'] = float(np.trapezoid((h['material_node'] == 0).astype(float), h['z_m']))
            records.append(rec)
            balance = storage - initial - top + flux_total
            checks.append(dict(population=population, case_id=case,
                nodal_volume_error_mm=err,
                max_balance_residual_mm=float(abs(balance).max()),
                late_flux_storage_difference_mm=rec['base_drainage_24_72_mm']-rec['drainage_24_72_mm'],
                input_difference_mm=top[0]-row['infiltration_mm']))
            assert abs(top[0]-row['infiltration_mm']) < .005, (case,top[0])
            assert abs(balance).max() < .2, (case, balance)
            if population == 'reference':
                ts = np.linspace(0, 72, 145)
                for t, s, q in zip(ts, np.interp(7200+ts*3600, tl[:, 0], tl[:, 16])*1000,
                                   -np.interp(7200+ts*3600, tl[:, 0], tl[:, 10])*1000):
                    histories.append(dict(case_id=case, order=row['order'], interface_m=row['interface_m'],
                        initial_head_m=row['initial_head_m'], infiltration_mm=row['infiltration_mm'],
                        time_h=t, storage_mm=s, drainage_since_input_start_mm=q))
                if row['infiltration_mm'] == 14.4:
                    blocks = node_blocks(native / 'NOD_INF.OUT')
                    for hour, target in zip(HOURS, query):
                        ix = np.argmin(abs(h['time_s']-target))
                        matching = [b[np.argsort(-b[:, 1])] for t,b in blocks if abs(t-target)<.1 and len(b)==401]
                        a = min(matching, key=lambda b: np.max(abs(b[:, 3]-h['theta_node'][ix])))
                        assert np.max(abs(a[:, 3]-h['theta_node'][ix])) < 1e-8
                        for j, z in enumerate(h['z_m']):
                            profiles.append(dict(case_id=case, order=row['order'],initial_head_m=row['initial_head_m'],
                                interface_m=row['interface_m'], time_h=hour, depth_m=z,
                                theta=a[j,3], head_m=a[j,2], conductivity_m_s=a[j,4],
                                downward_flux_mm_h=-a[j,6]*3.6e6, material=int(h['material_node'][j])))
        print(population, len(manifest), 'verified', flush=True)
    cases = pd.DataFrame(records)
    cases.to_csv(OUT / 'hydraulic_cases.csv', index=False)
    pd.DataFrame(histories).to_csv(OUT / 'reference_histories.csv', index=False)
    prof=pd.DataFrame(profiles)
    prof.to_csv(OUT / 'reference_profiles.csv', index=False)
    layer_rows=[]
    for (case,hour),g in prof.groupby(['case_id','time_h']):
        g=g.sort_values('depth_m');boundary=g.interface_m.iloc[0]
        for name,part in [('upper',g[g.depth_m<=boundary]),('lower',g[g.depth_m>=boundary])]:
            layer_rows.append(dict(case_id=case,time_h=hour,layer=name,
                order=int(g.order.iloc[0]),initial_head_m=g.initial_head_m.iloc[0],interface_m=boundary,
                storage_mm=float(np.trapezoid(part.theta,part.depth_m)*1000)))
    pd.DataFrame(layer_rows).to_csv(OUT/'layer_storage.csv',index=False)
    check = pd.DataFrame(checks)
    check.to_csv(OUT / 'water_balance_checks.csv', index=False)
    targets = ['initial_storage_mm','storage_0h_mm','storage_24h_mm','storage_72h_mm','drainage_24_72_mm','base_drainage_start_72h_mm']
    cases.groupby(['population','infiltration_mm','initial_head_m'])[targets].mean().to_csv(OUT / 'wetness_summary.csv')
    ref = cases[cases.population=='reference']
    key = ['order','interface_m','initial_head_m']
    zero = ref[ref.infiltration_mm==0].set_index(key)
    contrasts=[]
    for row in ref[ref.infiltration_mm>0].to_dict('records'):
        z=zero.loc[tuple(row[k] for k in key)]
        assert abs(row['initial_storage_mm']-z.initial_storage_mm)<1e-9
        result={k:row[k] for k in ['case_id']+key+['infiltration_mm']}
        for target in targets:
            result['delta_'+target]=row[target]-z[target]
        result['input_balance_residual_mm']=result['delta_storage_72h_mm']+result['delta_base_drainage_start_72h_mm']-row['infiltration_mm']
        contrasts.append(result)
    contrast=pd.DataFrame(contrasts)
    contrast.to_csv(OUT/'input_contrasts.csv', index=False)
    contrast.groupby(['infiltration_mm','initial_head_m']).mean(numeric_only=True).to_csv(OUT/'input_summary.csv')
    depth=cases[cases.population=='depth']
    depth.groupby(['order','initial_head_m','interface_m'])[targets].agg(['mean','min','max']).to_csv(OUT/'depth_summary.csv')
    paired=[]
    for (cluster,head),g in depth.groupby(['cluster_id','initial_head_m']):
        g=g.sort_values('interface_m');a,b=g.iloc[0],g.iloc[-1]
        paired.append(dict(cluster_id=cluster,order=a.order,initial_head_m=head,
                          **{'delta_'+t:b[t]-a[t] for t in targets}))
    pairs=pd.DataFrame(paired, columns=['cluster_id','order','initial_head_m']+['delta_'+t for t in targets])
    pairs.to_csv(OUT/'depth_contrasts.csv',index=False)
    pairs.groupby(['order','initial_head_m']).mean(numeric_only=True).to_csv(OUT/'depth_contrast_summary.csv')
    reversed_pairs=[]
    for row in ref[ref.order==0].to_dict('records'):
        other=ref[(ref.order==1)&np.isclose(ref.interface_m,1-row['interface_m'])&
                  (ref.initial_head_m==row['initial_head_m'])&(ref.infiltration_mm==row['infiltration_mm'])].iloc[0]
        reversed_pairs.append(dict(case_id=row['case_id'],other_case=other.case_id,
            initial_storage_difference_mm=other.initial_storage_mm-row['initial_storage_mm']))
    pd.DataFrame(reversed_pairs).to_csv(OUT/'reversed_order_discretization.csv',index=False)
    report=dict(cases=len(cases),maximum_balance_residual_mm=float(check.max_balance_residual_mm.max()),
        maximum_input_difference_mm=float(check.input_difference_mm.abs().max()),
        maximum_late_flux_storage_difference_mm=float(check.late_flux_storage_difference_mm.abs().max()),
        maximum_input_contrast_balance_residual_mm=float(contrast.input_balance_residual_mm.abs().max()),
        maximum_reversed_initial_storage_difference_mm=max(abs(x['initial_storage_difference_mm']) for x in reversed_pairs),
        times='Hours after end of the two-hour input window; zero-input controls use the same clock.',
        drainage='Positive downward cumulative basal flux; late drainage also calculated from nodal storage.',
        interpretation='Input contrasts compare total hydraulic responses, not water-source fractions. Depth contrasts also change material proportions and initial water inventory.')
    (OUT/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    (OUT/'source_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    print(contrast.groupby(['infiltration_mm','initial_head_m'])[['delta_storage_72h_mm','delta_base_drainage_start_72h_mm','delta_drainage_24_72_mm']].mean().to_string())


if __name__ == '__main__':
    main()
