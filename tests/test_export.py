# -*- coding: utf-8 -*-
"""``sarcasm.export`` — the boundary every result crosses on its way out.

Synthetic records exercise the writers without file IO; the ``BatchExport``
round trip runs on the 20 kPa reference store (never ``restart=True`` on it).
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import pytest

from sarcasm import SarcAsM
from sarcasm.export import BatchExport, Export

from .test_grouped_motion import _fake_structure


def _records(n=3, T=1500):
    """Per-group records as ``get_motion_dict_per_group`` shapes them, with a
    per-frame ``time`` vector longer than numpy's 1000-element print threshold."""
    time = np.arange(T) * 0.01
    return [{'file_name': 'movie.tif', 'pixelsize': 0.1, 'frametime': 0.01, 'n_stack': T,
             'time': time, 'timestamps': list(time), 'user_info': {'note': 'x'},
             'kind': 'pool', 'group_id': g, 'beating_rate': [1.0 + g], 'contr_max': 0.2 * g,
             'tif_name': 'movie.tif'} for g in range(n)]


# ---------------------------------------------------------------------------
# tabular cleaning
# ---------------------------------------------------------------------------

def test_tabular_frame_drops_per_frame_metadata_and_flattens_singletons():
    df = Export.tabular_frame(_records())
    for col in ('time', 'timestamps', 'user_info'):
        assert col not in df.columns
    assert df['beating_rate'].tolist() == [1.0, 2.0, 3.0]          # 1-element lists collapsed
    assert df['pixelsize'].tolist() == [0.1] * 3                   # scalar metadata kept
    assert list(df['group_id']) == [0, 1, 2]


def test_tabular_frame_accepts_a_dataframe():
    df = pd.DataFrame.from_records(_records())
    out = Export.tabular_frame(df)
    assert 'time' not in out.columns and len(out) == 3


def test_export_emits_no_pandas_deprecation_warning(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter('error', FutureWarning)
        warnings.simplefilter('error', DeprecationWarning)
        Export.write_records(str(tmp_path / 'r.csv'), _records(), 'csv')
        Export.tabular_frame(_records())


# ---------------------------------------------------------------------------
# write_records / write_dict
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('fmt', ['csv', '.csv', 'xlsx', '.xlsx'])
def test_write_records_tabular(tmp_path, fmt):
    out = tmp_path / f"records.{fmt.lstrip('.')}"
    Export.write_records(str(out), _records(), fmt)
    df = pd.read_csv(out) if 'csv' in fmt else pd.read_excel(out)
    assert len(df) == 3
    assert 'time' not in df.columns                                # not stringified/truncated
    assert df['beating_rate'].tolist() == [1.0, 2.0, 3.0]
    assert not any('...' in str(v) for v in df.iloc[0].values)


def test_write_records_json_keeps_metadata(tmp_path):
    out = tmp_path / 'records.json'
    Export.write_records(str(out), _records(), 'json')
    loaded = json.loads(out.read_text())
    assert len(loaded) == 3
    assert len(loaded[0]['time']) == 1500                          # JSON is lossless


def test_write_records_rejects_unknown_format(tmp_path):
    with pytest.raises(ValueError, match='Unsupported'):
        Export.write_records(str(tmp_path / 'r.txt'), _records(), 'txt')


def _structure_dict(T=5):
    return {'file_name': 'movie.tif', 'pixelsize': 0.1, 'frametime': 0.01, 'n_stack': T,
            'time': np.arange(T) * 0.01,
            'structure.sarcomere.length_mean': np.linspace(1.8, 1.9, T),
            'structure.cell.mask_area': 123.0,
            'structure.myofibril.length': [np.array([1.0, 2.0]), np.array([3.0])] + [np.array([])] * (T - 2)}


@pytest.mark.parametrize('fmt', ['csv', 'xlsx'])
def test_write_dict_tabular_is_framewise_without_metadata(tmp_path, fmt):
    out = tmp_path / f'structure.{fmt}'
    Export.write_dict(str(out), _structure_dict(), fmt)
    df = pd.read_csv(out, index_col=0) if fmt == 'csv' else pd.read_excel(out, index_col=0)
    assert list(df.columns) == [f'frame_{i}' for i in range(5)]
    assert 'time' not in df.index and 'pixelsize' not in df.index
    assert df.loc['structure.cell.mask_area'].tolist() == [123.0] * 5
    # ragged per-object distributions collapse to a per-frame mean
    assert df.loc['structure.myofibril.length'].iloc[0] == pytest.approx(1.5)


def test_write_dict_json_summary_and_raw(tmp_path):
    out = tmp_path / 's.json'
    Export.write_dict(str(out), _structure_dict(), 'json')
    summary = json.loads(out.read_text())
    assert summary['structure.cell.mask_area'] == 123.0
    assert len(summary['structure.sarcomere.length_mean']) == 5
    raw = tmp_path / 'raw.json'
    Export.write_dict(str(raw), _structure_dict(), 'json', raw=True)
    assert len(json.loads(raw.read_text())['structure.myofibril.length'][0]) == 2
    with pytest.raises(ValueError, match='JSON'):
        Export.write_dict(str(tmp_path / 'raw.csv'), _structure_dict(), 'csv', raw=True)


# ---------------------------------------------------------------------------
# per-group records from a synthetic analysis
# ---------------------------------------------------------------------------

def test_motion_records_collapse_per_cycle_arrays():
    sarc = _fake_structure(n_tracks=6)
    sarc.file_path = 'synthetic.tif'
    sarc.metadata.to_dict = lambda: {'file_name': 'synthetic.tif', 'frametime': 0.01}
    sarc.analyze_track_motion(by='mband', reference_frame=0)
    recs = Export.get_motion_dict_per_group(sarc, kind='mband', condition='ctrl')
    assert [r['group_id'] for r in recs] == [0, 1]
    for r in recs:
        assert r['kind'] == 'mband' and r['condition'] == 'ctrl'
        assert np.isscalar(r['contr_max']) or r['contr_max'] is None or np.ndim(r['contr_max']) == 0
    with pytest.raises(ValueError, match="No 'loi'"):
        Export.get_motion_dict_per_group(sarc, kind='loi')


# ---------------------------------------------------------------------------
# BatchExport
# ---------------------------------------------------------------------------

def test_batch_export_requires_collected_data(tmp_path):
    be = BatchExport([], folder=str(tmp_path))
    assert be.data is None
    with pytest.raises(ValueError, match='get_data'):
        be.export_data(str(tmp_path / 'x.csv'), '.csv')
    with pytest.raises(ValueError, match='get_data'):
        be.save_data()
    with pytest.raises(FileExistsError):
        be.load_motion_data()
    with pytest.raises(FileExistsError):
        BatchExport([], folder=str(tmp_path), load_data=True)


class TestBatchExportRoundTrip:

    @pytest.fixture(scope='class')
    def batch(self, motion_file_path_class, tmp_path_factory):
        s = SarcAsM(motion_file_path_class)
        if 'motion.tracks.slen' not in s.data:
            pytest.skip('20 kPa store has no tracks')
        s.analyze_track_motion(by='pool')            # (re)writes the per-kind provenance keys
        folder = tmp_path_factory.mktemp('batch')
        return BatchExport([motion_file_path_class], folder=str(folder), experiment='exp',
                           stiffness=lambda f: 'soft' if '20kPa' in f else 'stiff')

    def test_motion_table_round_trip(self, batch, tmp_path):
        batch.get_motion_data()
        df = batch.data
        assert len(df) > 0 and set(df['kind']).issubset({'pool', 'mband', 'myofibril', 'domain', 'loi', 'custom'})
        assert (df['stiffness'] == 'soft').all() and (df['experiment'] == 'exp').all()
        for fmt in ('.csv', '.xlsx'):
            out = tmp_path / f'motion{fmt}'
            batch.export_data(str(out), fmt)
            back = pd.read_csv(out) if fmt == '.csv' else pd.read_excel(out)
            assert len(back) == len(df) and 'time' not in back.columns
        fresh = BatchExport([], folder=batch.folder)
        fresh.load_motion_data()
        assert len(fresh.data) == len(df)

    def test_structure_table_round_trip(self, batch):
        batch.get_data()
        assert len(batch.data) == 1
        assert 'structure.sarcomere.slen_mean' in batch.data.columns
        fresh = BatchExport([], folder=batch.folder, load_data=True)
        assert fresh.data.shape == batch.data.shape
