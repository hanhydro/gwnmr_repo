from pathlib import Path
import zipfile

STUDY = Path(__file__).resolve().parents[1]

if __name__ == '__main__':
    with zipfile.ZipFile(STUDY / 'data/redistribution_20261112/native_outputs.zip') as archive:
        archive.extractall(STUDY)
