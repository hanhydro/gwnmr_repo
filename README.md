# Drainage and surface-NMR responses to increasing infiltration in layered soils

Code and input data for the calculations, figures, and tables of the manuscript with this title.

## Requirements

- Python 3.11 with the packages in `requirements.txt`.
- Helvetica at `/System/Library/Fonts/Helvetica.ttc` (macOS) for the figures. No font is distributed.

## Reproduce

```sh
python3 -m pip install -r requirements.txt
sh reproduce.sh
```

`reproduce.sh` unpacks the archived HYDRUS-1D files into `work/` and the nodal arrays into `data/redistribution_20261112/hydraulics/`, recalculates all summaries, and writes the figures to `figures/`. It took about one minute on the tested computer (macOS, Python 3.11.8). Set `PYTHON` to use a different interpreter.

## Outputs

| Item | Script | Output |
|---|---|---|
| Figure 1 | `analysis/figure_redesign_20261116/figure1.py` | `figures/study_conceptual_overview` |
| Figure 2 | `analysis/nsg_revision_20261117/figure2_decomposition.py` | `figures/input_response_configurations_decomposition` |
| Figures 3 and 4 | `analysis/input_response_20261112/figures.py` | `figures/input_response_histories`, `figures/input_response_pores_signals` |
| Figure S1 | `analysis/redistribution_20261112/figures.py` | `figures/redistribution_depth_support` |
| Figure S2 | `analysis/input_response_20261112/material_figure.py` | `figures/redistribution_pore_support` |
| Table S1 | `analysis/redistribution_20261112/summarize.py`, `analysis/redistribution_20261112/check_pore_numerics.py`, `analysis/input_response_20261112/pores.py` | `data/redistribution_20261112/redistribution_values.tex`, `data/redistribution_20261112/signal_numerical_comparisons.csv`, `data/input_response_20261112/pore_verification.json` |
| Table S2 | `analysis/study_overview_20261116/settings.py` | `data/study_overview_20261116/hydraulic_settings_table.tex` |
| Table S3 | `analysis/input_response_20261112/synthesis.py` | `data/input_response_20261112/table_individual_inputs.tex` |
| Table S4 | `analysis/mathematical_generalization_20261116/evaluate.py` | `data/mathematical_generalization_20261116/basal_state_integrals.csv` |
| Table S5 | `analysis/input_response_20261112/update_support.py` | `data/input_response_20261112/table_signal_precision.tex` |

Intermediate results are written to the same `data/` folders. Time is in hours after the two-hour input ended. Depth is in m, water content in m³ m⁻³, storage and drainage in mm, pulse moment in A ms, and voltage in nV.

## Input data

| File | Contents |
|---|---|
| `data/redistribution_20261112/native_outputs_*.zip` | HYDRUS-1D inputs (`SELECTOR.IN`, `PROFILE.DAT`, `ATMOSPH.IN`), outputs (`T_LEVEL.OUT`, `BALANCE.OUT`, `I_CHECK.OUT`), and run settings for all 646 runs, split into parts of less than 14 MB |
| `data/redistribution_20261112/manifest.csv` | Layer order, interface depth, initial head, and input of each run |
| `data/redistribution_20261112/hydraulic_arrays_*.zip`, `data/input_response_20261112/hydraulics/` | Nodal water content and pressure head of each run |
| `data/observation_kernel_full_100.npz` | Surface-NMR sensitivity kernels from full-field Bloch excitation, 100 ohm m |
| `data/dephasing_model.npz` | Static frequency-offset integration |
| `data/input_response_20261112/spatial_*.csv` | Refined upper-meter electromagnetic integration (Table S5) |
| `data/input_response_20261112/previous_pore_moments.csv.gz` | Filled-interval pore integration used in the quadrature comparison |
| `data/physical_recovery_20260929/matched_capillary_water_control.*`, `data/figure_revision_20261022/filled_pore_cumulative_volume.csv` | Material pore distributions at equal capillary water (Figure S2) |

## Calculations not repeated here

The water-flow runs used HYDRUS-1D ([source](https://github.com/phydrus/source_code), commit `b984da66a5b86f46f6b576dea0bf200d0c346f82`) through Phydrus 0.2.0. The archive contains the exact input files, so each run can be repeated with a HYDRUS-1D executable. The sensitivity kernels and the refined electromagnetic integration used MRSmatlab fields ([Müller-Petke et al., 2016](https://doi.org/10.1190/geo2015-0461.1)) and a full-field Bloch excitation calculation. Their results are supplied as input data.

## License

MIT. See `LICENSE`.
