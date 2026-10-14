"""Populate DATA_ROOT with the OCB datasets from the official repo (idempotent).

Copies Ckt-Bench-101 as-is and extracts the Ckt-Bench-301 zips (skipping macOS metadata).
The repository copy under external/CktGNN is left untouched.

Usage:  .venv/Scripts/python.exe scripts/setup_data.py
"""
import os
import shutil
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import REPO, DATASETS, RESULTS, BENCH101, BENCH301  # noqa: E402


def main():
    for d in (BENCH101, BENCH301, RESULTS):
        os.makedirs(d, exist_ok=True)
    src101 = os.path.join(REPO, 'OCB', 'CktBench101')
    for name in os.listdir(src101):
        dst = os.path.join(BENCH101, name)
        if not os.path.exists(dst):
            shutil.copy2(os.path.join(src101, name), dst)
            print('copied', dst)
    src301 = os.path.join(REPO, 'OCB', 'CktBench301')
    for name in os.listdir(src301):
        path = os.path.join(src301, name)
        if name.endswith('.zip'):
            with zipfile.ZipFile(path) as z:
                for info in z.infolist():
                    if info.filename.startswith('__MACOSX') or os.path.exists(os.path.join(BENCH301, info.filename)):
                        continue
                    z.extract(info, BENCH301)
                    print('extracted', os.path.join(BENCH301, info.filename))
        elif not os.path.exists(os.path.join(BENCH301, name)):
            shutil.copy2(path, os.path.join(BENCH301, name))
            print('copied', os.path.join(BENCH301, name))
    print('datasets ready under', DATASETS)


if __name__ == '__main__':
    main()
