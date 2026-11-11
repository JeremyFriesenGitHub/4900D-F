import numpy as np

from arap.rotations import fit_rotations

from conftest import random_rotation


def test_recovers_rotation_from_noise_free_covariance(rng):
    # S = sum_j e_j (R e_j)^T  ->  argmax Tr(R S) is R.
    Rs = np.stack([random_rotation(rng) for _ in range(20)])
    S = []
    for R in Rs:
        E = rng.normal(size=(6, 3))
        S.append(E.T @ (E @ R.T))
    out = fit_rotations(np.stack(S))
    assert np.abs(out - Rs).max() < 1e-10


def test_output_is_always_a_proper_rotation(rng):
    S = rng.normal(size=(500, 3, 3))
    R = fit_rotations(S)
    I = np.broadcast_to(np.eye(3), R.shape)
    assert np.abs(np.swapaxes(R, 1, 2) @ R - I).max() < 1e-10
    assert np.abs(np.linalg.det(R) - 1.0).max() < 1e-10


def test_reflection_case_is_optimal(rng):
    # For covariances whose polar factor is a reflection, compare Tr(R S)
    # with many random rotations: none may do better.
    S = np.diag([3.0, 2.0, -1.0])
    Q1, Q2 = random_rotation(rng), random_rotation(rng)
    S = Q1 @ S @ Q2
    R = fit_rotations(S[None])[0]
    best = np.trace(R @ S)
    trials = [np.trace(random_rotation(rng) @ S) for _ in range(5000)]
    assert best >= max(trials) - 1e-12
    # the known optimum: sum of singular values with the smallest negated
    assert np.isclose(best, 3.0 + 2.0 - 1.0)


def test_planar_covariance_gives_rotation(rng):
    # All edges in a plane: S has rank 2, the result must still be a rotation.
    E = rng.normal(size=(5, 3))
    E[:, 2] = 0.0
    S = E.T @ E
    R = fit_rotations(S[None])[0]
    assert np.isclose(np.linalg.det(R), 1.0)
