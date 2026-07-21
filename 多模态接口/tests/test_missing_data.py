import numpy as np

from geo_adapter.preprocess.missing_data import interpolate_short_gaps, max_consecutive_false


def test_short_gap_interpolates_and_long_gap_remains_missing() -> None:
    values = np.array([1.0, np.nan, np.nan, 4.0, np.nan, np.nan, np.nan, np.nan, 9.0])
    cleaned, valid, interpolated = interpolate_short_gaps(values, max_gap_samples=2)
    assert np.allclose(cleaned[1:3], [2.0, 3.0])
    assert interpolated.tolist() == [False, True, True, False, False, False, False, False, False]
    assert np.isnan(cleaned[4:8]).all()
    assert not (valid & interpolated).any()
    assert max_consecutive_false(valid) == 4


def test_edge_gap_is_not_extrapolated() -> None:
    cleaned, _, interpolated = interpolate_short_gaps(np.array([np.nan, 1.0, 2.0, np.nan]), 3)
    assert np.isnan(cleaned[[0, 3]]).all()
    assert not interpolated.any()

