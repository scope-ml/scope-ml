"""Tests for the single-point spike filter in tools/generate_features_rubin.py."""

import importlib.util
import pathlib

import numpy as np
import pytest


@pytest.fixture(scope="module")
def gfr():
    path = pathlib.Path(__file__).parent.parent / "tools" / "generate_features_rubin.py"
    spec = importlib.util.spec_from_file_location("gfr", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _flat(n=40, mag=17.0, scatter=0.02, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    m = mag + rng.normal(0, scatter, n)
    return t, m, np.zeros(n, dtype=int)


def test_single_spike_is_removed(gfr):
    t, m, b = _flat()
    m[20] += 3.0
    keep = gfr.spike_mask(t, m, b, floor=1.0)
    assert not keep[20]
    assert keep.sum() == len(m) - 1


def test_eclipse_is_kept(gfr):
    # Detached eclipsing binary: flat baseline with a 2 mag eclipse covered by
    # several consecutive points. A 5-sigma MAD clip removes all of them.
    t, m, b = _flat(n=60, scatter=0.01)
    m[30:34] += np.array([1.0, 2.0, 2.0, 1.0])
    keep = gfr.spike_mask(t, m, b, floor=1.0)
    assert keep.all()
    sigma = 1.4826 * np.median(np.abs(m - np.median(m)))
    assert (np.abs(m - np.median(m)) >= 5.0 * sigma)[30:34].all()


def test_spikes_at_the_ends_are_removed(gfr):
    t, m, b = _flat()
    m[0] -= 2.5
    m[-1] += 2.5
    keep = gfr.spike_mask(t, m, b, floor=1.0)
    assert not keep[0] and not keep[-1]
    assert keep[1:-1].all()


def test_bands_are_filtered_separately(gfr):
    # Two bands interleaved in time, 1.5 mag apart: no point is a spike within
    # its own band, so nothing may be removed.
    n = 40
    t = np.arange(n, dtype=float)
    b = np.arange(n) % 2
    m = np.where(b == 0, 17.0, 18.5)
    keep = gfr.spike_mask(t, m, b, floor=1.0)
    assert keep.all()


def test_infinite_floor_keeps_everything(gfr):
    t, m, b = _flat()
    m[10] += 5.0
    assert gfr.spike_mask(t, m, b, floor=np.inf).all()


def test_input_order_does_not_matter(gfr):
    t, m, b = _flat(seed=3)
    m[15] += 3.0
    order = np.random.default_rng(1).permutation(len(t))
    keep_sorted = gfr.spike_mask(t, m, b, floor=1.0)
    keep_shuffled = gfr.spike_mask(t[order], m[order], b[order], floor=1.0)
    assert np.array_equal(keep_sorted[order], keep_shuffled)


def test_short_bands_are_untouched(gfr):
    t = np.array([0.0, 1.0])
    m = np.array([17.0, 25.0])
    b = np.array([0, 0])
    assert gfr.spike_mask(t, m, b, floor=1.0).all()
