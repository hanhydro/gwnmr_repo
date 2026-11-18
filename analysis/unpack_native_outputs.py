from pathlib import Path
import zipfile

STUDY = Path(__file__).resolve().parents[1]

if __name__ == '__main__':
    folder = STUDY / 'data/redistribution_20261112'
    for part in sorted(folder.glob('hydraulic_arrays_*.zip')) + sorted(folder.glob('native_outputs_*.zip')):
        with zipfile.ZipFile(part) as archive:
            archive.extractall(STUDY)
