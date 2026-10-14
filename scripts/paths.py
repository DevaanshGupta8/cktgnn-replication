"""Shared locations.

Code and reports live in the project folder; datasets, checkpoints and results live under
DATA_ROOT (default D:\\CktGNN_data, override with the CKTGNN_DATA environment variable).
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.join(ROOT, 'external', 'CktGNN')

DATA_ROOT = os.environ.get('CKTGNN_DATA', r'D:\CktGNN_data')
DATASETS = os.path.join(DATA_ROOT, 'datasets')
RESULTS = os.path.join(DATA_ROOT, 'results')
BENCH101 = os.path.join(DATASETS, 'CktBench101')
BENCH301 = os.path.join(DATASETS, 'CktBench301')
