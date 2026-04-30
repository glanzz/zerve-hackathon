from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional
import pandas as pd
from config.config import CFG
_log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
try:
    from loguru import logger as _L
    _info = lambda m: _L.info(m)
    _warn = lambda m: _L.warning(m)
except ImportError:
    _info = lambda m: _log.info(m)
    _warn = lambda m: _log.warning(m)

def load_raw_events(path=None, drop_banned=True):
    _p = Path(path or CFG.RAW_EVENTS_PATH)
    if not _p.exists():
        raise FileNotFoundError('Raw events not found: ' + str(_p))
    _info('Loading from ' + str(_p))
    _df = pd.read_parquet(_p)
    _info('Loaded {:,} rows {:,} users'.format(len(_df), _df[CFG.USER_ID_COL].nunique()))
    if str(_df[CFG.TIMESTAMP_COL].dtype) == 'object':
        _df[CFG.TIMESTAMP_COL] = pd.to_datetime(_df[CFG.TIMESTAMP_COL], utc=True, errors='coerce')
    elif getattr(_df[CFG.TIMESTAMP_COL].dtype, 'tz', None) is None:
        _df[CFG.TIMESTAMP_COL] = _df[CFG.TIMESTAMP_COL].dt.tz_localize('UTC')
    _n_null = int(_df[CFG.TIMESTAMP_COL].isna().sum())
    if _n_null:
        _warn('Dropping {:,} null-ts rows'.format(_n_null))
        _df = _df.dropna(subset=[CFG.TIMESTAMP_COL])
    if drop_banned:
        _before = len(_df)
        for _bev in CFG.BANNED_EVENT_NAMES:
            _msk = _df[CFG.EVENT_TYPE_COL] == _bev
            _nev = int(_msk.sum())
            if _nev:
                _info('Dropped {:,} rows event={!r}'.format(_nev, _bev))
                _df = _df[~_msk]
        _nd = _before - len(_df)
        _info('Banned filter: {:,} dropped ({:.2f}%). Remaining: {:,}'.format(_nd, _nd/_before*100, len(_df)))
    return _df.reset_index(drop=True)

def load_feature_store(path=None):
    _p = Path(path or CFG.FEATURE_STORE_PATH)
    if not _p.exists(): raise FileNotFoundError('Feature store not found: ' + str(_p))
    return pd.read_parquet(_p)

def load_labels(path=None):
    _p = Path(path or CFG.LABELS_PATH)
    if not _p.exists(): raise FileNotFoundError('Labels not found: ' + str(_p))
    return pd.read_parquet(_p)

def load_splits(path=None):
    _p = Path(path or CFG.SPLITS_PATH)
    if not _p.exists(): raise FileNotFoundError('Splits not found: ' + str(_p))
    return pd.read_parquet(_p)