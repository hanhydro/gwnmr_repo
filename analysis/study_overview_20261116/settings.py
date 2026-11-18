from pathlib import Path
import csv
import hashlib
import json
from collections import Counter
import numpy as np

STUDY = Path(__file__).resolve().parents[2]
OUT = STUDY / 'data/study_overview_20261116'


def following(lines, prefix):
    return lines[next(i for i, line in enumerate(lines) if line.startswith(prefix)) + 1].split()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with (STUDY / 'data/redistribution_20261112/manifest.csv').open() as f:
        manifest = {r['case_id']: r for r in csv.DictReader(f)}
    rows = []
    groups = [('standard', 'redistribution_20261112'), ('half_hour', 'input_response_20261112'),
              ('pilot5', 'redistribution_20261112'), ('step60', 'redistribution_20261112'),
              ('mesh800', 'redistribution_20261112')]
    for tag, source in groups:
        for path in sorted((STUDY / f'work/{source}/hydrus/{tag}').glob('*/configuration.json')):
            c = json.loads(path.read_text())
            selector = path.with_name('SELECTOR.IN')
            lines = selector.read_text().splitlines()
            time = list(map(float, following(lines, 'dt dtMin')))
            flow = list(map(float, following(lines, 'MaxIt')))
            initial, end = map(float, following(lines, 'tInit tMax'))
            profile = path.with_name('PROFILE.DAT').read_text().splitlines()
            nodes = int(profile[2].split()[0])
            positions = np.array([float(x.split()[1]) for x in profile[3:3+nodes]])
            heads = np.array([float(x.split()[2]) for x in profile[3:3+nodes]])
            assert nodes == c['grid_cells'] + 1
            assert np.allclose(np.diff(positions), -1/c['grid_cells'])
            assert np.allclose(heads, c['initial_head_m'])
            assert time[0] == c['metadata']['phydrus_time']['dt']
            assert time[1] == c['metadata']['phydrus_time']['dtmin']
            assert time[2] == c['metadata']['phydrus_time']['dtmax']
            assert flow == [c['metadata']['phydrus_flow'][key] for key in ['maxit', 'tolth', 'tolh']]
            assert following(lines, 'ha  hb') == ['1e-06', '1e-06']
            assert following(lines, 'iModel') == ['0', '0']
            assert following(lines, 'BotInf')[2] == 't'
            row = dict(case_id=c['name'], series=tag,
                       population=manifest[c['name']]['split'] if tag == 'standard' else tag,
                       elements=c['grid_cells'], spacing_m=1/c['grid_cells'],
                       initial_step_s=time[0], minimum_step_s=time[1], maximum_step_s=time[2],
                       step_increase_factor=time[3], step_decrease_factor=time[4],
                       lower_iteration_threshold=int(time[5]), upper_iteration_threshold=int(time[6]),
                       maximum_iterations=int(flow[0]), theta_tolerance=flow[1], head_tolerance_m=flow[2],
                       calculation_end_s=end, analyzed_end_s=266400,
                       source_sha256=hashlib.sha256(selector.read_bytes()).hexdigest(),
                       profile_sha256=hashlib.sha256(path.with_name('PROFILE.DAT').read_bytes()).hexdigest())
            rows.append(row)
    with (OUT / 'hydraulic_settings_by_case.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    keys = ['population', 'elements', 'maximum_step_s', 'theta_tolerance', 'head_tolerance_m']
    counts = Counter(tuple(row[k] for k in keys) for row in rows)
    summary = [dict(zip(keys, values), count=count) for values, count in sorted(counts.items())]
    assert sum(r['count'] for r in summary if r['population'] == 'mechanism') == 54
    assert len(rows) == 646
    q = np.load(STUDY / 'data/observation_kernel_full_100.npz')['q_ams']
    shared = {key: sorted(set(row[key] for row in rows)) for key in
              ['initial_step_s', 'minimum_step_s', 'maximum_iterations', 'step_increase_factor',
               'step_decrease_factor', 'lower_iteration_threshold', 'upper_iteration_threshold', 'calculation_end_s']}
    c = json.loads((STUDY / 'work/redistribution_20261112/hydrus/standard/mechanism_o0_d0_h1_r144/configuration.json').read_text())
    report = dict(native_input_cases_checked=len(rows), groups=summary, shared=shared,
                  reference_material_parameters=c['soils'],
                  output_times_hours_after_input=[0, 6, 24, 48, 72],
                  original_profile_times_hours_after_input=[(v-7200)/3600 for v in c['metadata']['phydrus_print_times_s']],
                  pulse_moments_ams=q.tolist(), report_end_h=72,
                  calculation_end_h_after_input=(shared['calculation_end_s'][0]-7200)/3600,
                  phydrus_version_from_native_header='0.2.0')
    exe = STUDY.parent / 'external/phydrus/examples/hydrus'
    if exe.exists(): report['hydrus_executable_sha256'] = hashlib.sha256(exe.read_bytes()).hexdigest()
    (OUT / 'verified_settings.json').write_text(json.dumps(report, indent=2) + '\n')
    labels = {'mechanism':'Reference input comparisons', 'half_hour':'Selected detailed histories',
              'confirmation':'Varied interface depths', 'confirmation_depth':'Matched interface depths',
              'pilot5':'Step comparison', 'step60':'Step comparison', 'mesh800':'Mesh comparison'}
    order = ['mechanism', 'half_hour', 'confirmation', 'confirmation_depth', 'step60', 'pilot5', 'mesh800']
    tex = [r'\begin{table}[htbp]\centering\small',
           r'\caption{Hydraulic discretization and convergence settings. The tolerance columns give absolute water-content and pressure-head tolerances. Detailed histories repeat six reference cases. Step and mesh comparisons repeat reference cases at a 0.50 m interface.}',
           r'\label{tab:solver_settings}',
           r'\begin{tabular}{p{.28\textwidth}rrrrr}\toprule',
           r'Calculation & Runs & Elements & Maximum step (s) & $\theta$ & $h$ (m)\\\midrule']
    for population in order:
        for r in sorted([r for r in summary if r['population'] == population], key=lambda r:-r['maximum_step_s']):
            th = int(round(np.log10(r['theta_tolerance']))); hh = int(round(np.log10(r['head_tolerance_m'])))
            tex.append(f"{labels[population]} & {r['count']} & {r['elements']} & {r['maximum_step_s']:g} & $10^{{{th}}}$ & $10^{{{hh}}}$" + r'\\')
    tex.extend([r'\bottomrule\end{tabular}\end{table}', ''])
    (OUT / 'hydraulic_settings_table.tex').write_text('\n'.join(tex))
    print(json.dumps(dict(checked=len(rows), groups=summary, shared=shared), indent=2))


if __name__ == '__main__': main()
