"""Best-fit rotations for the ARAP local step (paper Section 2.1)."""

import numpy as np


def fit_rotations(covariances):
    """Closest rotation for a stack of 3x3 covariance matrices.

    For ``S_i = U_i Sigma_i V_i^T`` the rotation maximising
    ``Tr(R_i S_i)`` is ``R_i = V_i U_i^T`` (paper Eq. 6). If that matrix is a
    reflection (det < 0), the column of ``U_i`` belonging to the smallest
    singular value is negated, which gives the optimal proper rotation.

    Parameters
    ----------
    covariances : (n, 3, 3) array

    Returns
    -------
    R : (n, 3, 3) array of rotations (det = +1)
    """
    S = np.asarray(covariances, dtype=np.float64)
    U, _, Vt = np.linalg.svd(S)
    V = np.swapaxes(Vt, 1, 2)
    R = V @ np.swapaxes(U, 1, 2)
    reflected = np.linalg.det(R) < 0.0
    if np.any(reflected):
        U = U.copy()
        # numpy returns singular values in descending order, so the last
        # column belongs to the smallest one.
        U[reflected, :, 2] *= -1.0
        R[reflected] = V[reflected] @ np.swapaxes(U[reflected], 1, 2)
    return R
