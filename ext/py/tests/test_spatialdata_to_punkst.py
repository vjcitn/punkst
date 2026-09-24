import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest.importorskip("spatialdata")

from make_synthetic_sdata import build  # noqa: E402
from spatialdata_to_punkst import convert  # noqa: E402


def _read_output(path):
    with open(path) as fh:
        header = fh.readline()
    df = pd.read_csv(path, sep="\t", comment="#", header=None, names=["x", "y", "feature"])
    return header, df


def test_round_trip_identity_coordinate_system(tmp_path):
    zarr_path = tmp_path / "synthetic.zarr"
    truth = build(str(zarr_path), n_points=150, scale=2.0, seed=1)

    out_path = tmp_path / "transcripts.tsv"
    convert(str(zarr_path), "transcripts", str(out_path), coordinate_system="pixels")

    header, df = _read_output(out_path)
    assert header.startswith("#x\ty\tfeature")
    assert len(df) == len(truth)
    assert set(df["feature"]) <= set(truth["gene"].astype(str))

    truth_sorted = truth.sort_values(["x", "y"]).reset_index(drop=True)
    df_sorted = df.sort_values(["x", "y"]).reset_index(drop=True)
    np.testing.assert_allclose(df_sorted["x"], truth_sorted["x"], atol=1e-3)
    np.testing.assert_allclose(df_sorted["y"], truth_sorted["y"], atol=1e-3)


def test_coordinate_system_transform_is_applied(tmp_path):
    zarr_path = tmp_path / "synthetic.zarr"
    truth = build(str(zarr_path), n_points=100, scale=3.0, seed=2)

    out_path = tmp_path / "transcripts_microns.tsv"
    convert(str(zarr_path), "transcripts", str(out_path), coordinate_system="microns")

    _, df = _read_output(out_path)
    truth_sorted = truth.sort_values(["x", "y"]).reset_index(drop=True)
    df_sorted = df.sort_values(["x", "y"]).reset_index(drop=True)
    # "microns" coordinate system is a 3x scale of the raw "pixels" data
    np.testing.assert_allclose(df_sorted["x"], truth_sorted["x"] * 3.0, atol=1e-3)
    np.testing.assert_allclose(df_sorted["y"], truth_sorted["y"] * 3.0, atol=1e-3)


def test_ambiguous_coordinate_system_requires_explicit_choice(tmp_path):
    zarr_path = tmp_path / "synthetic.zarr"
    build(str(zarr_path), n_points=10, seed=3)

    out_path = tmp_path / "transcripts.tsv"
    with pytest.raises(ValueError):
        convert(str(zarr_path), "transcripts", str(out_path))


def test_unknown_points_key_raises(tmp_path):
    zarr_path = tmp_path / "synthetic.zarr"
    build(str(zarr_path), n_points=10, seed=4)

    out_path = tmp_path / "transcripts.tsv"
    with pytest.raises(KeyError):
        convert(str(zarr_path), "does_not_exist", str(out_path), coordinate_system="pixels")


def test_intrinsic_exports_stored_coordinates(tmp_path):
    zarr_path = tmp_path / "synthetic.zarr"
    truth = build(str(zarr_path), n_points=100, scale=3.0, seed=5)

    out_path = tmp_path / "transcripts_intrinsic.tsv"
    convert(str(zarr_path), "transcripts", str(out_path), coordinate_system="intrinsic")

    _, df = _read_output(out_path)
    truth_sorted = truth.sort_values(["x", "y"]).reset_index(drop=True)
    df_sorted = df.sort_values(["x", "y"]).reset_index(drop=True)
    np.testing.assert_allclose(df_sorted["x"], truth_sorted["x"], atol=1e-3)
    np.testing.assert_allclose(df_sorted["y"], truth_sorted["y"], atol=1e-3)


def test_unknown_coordinate_system_raises(tmp_path):
    zarr_path = tmp_path / "synthetic.zarr"
    build(str(zarr_path), n_points=10, seed=6)
    with pytest.raises(ValueError):
        convert(str(zarr_path), "transcripts", str(tmp_path / "o.tsv"), coordinate_system="nope")


def test_min_qv_filters_rows(tmp_path):
    zarr_path = tmp_path / "synthetic.zarr"
    truth = build(str(zarr_path), n_points=300, seed=7)

    out_path = tmp_path / "transcripts_q.tsv"
    convert(str(zarr_path), "transcripts", str(out_path),
            coordinate_system="pixels", min_qv=20)

    _, df = _read_output(out_path)
    assert len(df) == int((truth["qv"] >= 20).sum())
    assert 0 < len(df) < len(truth)
    assert list(pd.read_csv(out_path, sep="\t", comment="#", header=None).columns) == [0, 1, 2]
