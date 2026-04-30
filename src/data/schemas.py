from __future__ import annotations
import logging, pandas as pd
_log = logging.getLogger(__name__)
try:
    import pandera as pa
    _PANDERA_AVAILABLE = True
except ImportError:
    _PANDERA_AVAILABLE = False
def _sv(df, cols, name):
    miss = [c for c in cols if c not in df.columns]
    if miss: raise ValueError('[' + name + '] Missing: ' + str(miss))
    for c in cols:
        if df[c].isna().all(): raise ValueError('[' + name + '] ' + c + ' all-null')
    return df
_RE = ['person_id', 'timestamp', 'event']
_FS = ['person_id', 'user_cutoff_ts']
_LB = ['person_id', 'will_upgrade_in_7d', 'user_cutoff_ts']
def validate_raw_events(df): return _sv(df, _RE, 'RawEvents')
def validate_feature_store(df): return _sv(df, _FS, 'FeatureStore')
def validate_labels(df):
    _sv(df, _LB, 'Labels')
    b = df['will_upgrade_in_7d'].dropna()
    if not b.isin([0,1]).all(): raise ValueError('Labels: not 0/1')
    return df
def validate_prediction_input(df): return _sv(df, ['person_id'], 'PredInput')