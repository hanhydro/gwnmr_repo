from pathlib import Path
import sys
import numpy as np

STUDY = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(STUDY / 'analysis/physical_recovery_20260929'), str(STUDY / 'analysis')]
import signals
from extract import native_bytes


def current_parameters(row, tag, stored):
    path = STUDY / 'work/redistribution_20261112/hydrus' / tag / row['case_id'] / 'SELECTOR.IN'
    lines = native_bytes(path).decode().splitlines()
    start = next(i for i, line in enumerate(lines)
                 if 'thr' in line and 'ths' in line and 'Alfa' in line) + 1
    values = np.array([[float(x) for x in line.split()[:5]] for line in lines[start:start+2]])
    return values if int(row['order']) == 0 else values[::-1]


signals.actual_parameters = current_parameters
convert = signals.convert
operator = signals.operator
