import numpy as np
import pytest

from arap import selection as sel


def test_combine_sorts_and_keeps_targets_aligned(rng):
    V = rng.normal(size=(10, 3))
    idx, tgt = sel.combine(V, fixed=[7, 1], handles=[4, 0], handle_targets=[[1, 1, 1], [2, 2, 2]])
    assert idx.tolist() == [0, 1, 4, 7]
    assert np.array_equal(tgt[0], [2, 2, 2])
    assert np.array_equal(tgt[2], [1, 1, 1])
    assert np.array_equal(tgt[1], V[1])
    assert np.array_equal(tgt[3], V[7])


def test_combine_checks_target_count(rng):
    V = rng.normal(size=(10, 3))
    with pytest.raises(ValueError, match="shape"):
        sel.combine(V, fixed=[1], handles=[2, 0], handle_targets=np.zeros((3, 3)))


def test_combine_rejects_overlap(rng):
    V = rng.normal(size=(10, 3))
    with pytest.raises(ValueError):
        sel.combine(V, fixed=[1, 2], handles=[2], handle_targets=np.zeros((1, 3)))


def test_rigid_targets_rotate_about_pivot():
    P = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    out = sel.rigid_targets(P, sel.rotation("z", 90.0), pivot=[1.0, 0.0, 0.0], translation=[0, 0, 1])
    assert np.allclose(out, [[1, 0, 1], [1, 1, 1]])


def test_by_coordinate_uses_relative_thresholds():
    V = np.column_stack([np.zeros(11), np.zeros(11), np.linspace(-5, 5, 11)])
    assert sel.by_coordinate(V, "z", below=0.15).tolist() == [0, 1]
    assert sel.by_coordinate(V, "z", above=0.85).tolist() == [9, 10]
