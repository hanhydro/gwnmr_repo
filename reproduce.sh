set -e
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
mkdir -p figures
"$PY" analysis/unpack_native_outputs.py
"$PY" analysis/redistribution_20261112/extract.py
"$PY" analysis/redistribution_20261112/pore_responses.py
"$PY" analysis/redistribution_20261112/check_pore_numerics.py
"$PY" analysis/redistribution_20261112/summarize.py
"$PY" analysis/input_response_20261112/hydraulics.py
"$PY" analysis/input_response_20261112/pores.py
"$PY" analysis/input_response_20261112/check_pore_integration.py
"$PY" analysis/input_response_20261112/synthesis.py
"$PY" analysis/input_response_20261112/update_support.py
"$PY" analysis/mathematical_generalization_20261116/evaluate.py
"$PY" analysis/study_overview_20261116/settings.py
"$PY" analysis/figure_redesign_20261116/figure1.py
"$PY" analysis/input_response_20261112/figures.py
"$PY" analysis/nsg_revision_20261117/figure2_decomposition.py
"$PY" analysis/redistribution_20261112/figures.py
"$PY" analysis/input_response_20261112/material_figure.py
