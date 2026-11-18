from pathlib import Path
import json
import numpy as np
import pandas as pd
STUDY=Path(__file__).resolve().parents[2];DATA=STUDY/'data/input_response_20261112'

def main():
    temporal=pd.read_csv(DATA/'spatial_temporal_differences.csv')
    inputs=pd.read_csv(DATA/'spatial_input_differences.csv').pivot(index=['order','time_h'],columns='level',values='input_magnitude_difference_nV')
    layers=pd.read_csv(DATA/'spatial_layer_contributions.csv').pivot(index=['order','time_h'],columns='level',values='lower_fraction')
    table=[r'\begin{table}[htbp]\centering\small',r'\caption{Effect of upper-meter electromagnetic refinement at 21.24 A ms. Values are maximum absolute differences from the 0.005 m calculation. Temporal rows cover eight columns, including all three inputs for the two selected configurations. Input rows compare 28.8 with 14.4 mm in those two configurations.}',r'\label{tab:signal_precision}',r'\begin{tabular}{p{.56\textwidth}rr}\toprule',r'Quantity & 0.025 m & 0.0125 m\\\midrule']
    metrics=[]
    for label,func in [
        ('Temporal amplitude change (percentage points)',lambda l:temporal[temporal.level==l].amplitude_percent_difference.abs().max()),
        ('Normalized temporal curve (percentage points)',lambda l:temporal[temporal.level==l].normalized_temporal_curve_difference_pp.abs().max()),
        ('Input magnitude difference, all five times (nV)',lambda l:(inputs[l]-inputs[2]).abs().max()),
        ('Input magnitude difference, 24--72 h (nV)',lambda l:(inputs.loc[(slice(None),[24,48,72]),l]-inputs.loc[(slice(None),[24,48,72]),2]).abs().max()),
        ('Lower-layer projected fraction (percentage points)',lambda l:100*(layers[l]-layers[2]).abs().max())]:
        a,b=float(func(0)),float(func(1));table.append(f'{label} & {a:.5f} & {b:.5f}'+r'\\');metrics.append(dict(quantity=label,coarse_vs_fine=a,intermediate_vs_fine=b))
    table+=[r'\bottomrule\end{tabular}\end{table}']
    (DATA/'table_signal_precision.tex').write_text('\n'.join(table)+'\n')
    (DATA/'signal_precision_summary.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print(json.dumps(metrics,indent=2))
if __name__=='__main__':main()
