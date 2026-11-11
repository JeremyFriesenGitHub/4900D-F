import os

import numpy as np
import pytest

import geomproc
from arap.meshio import from_geomproc, load_mesh

MESH_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "meshes")


def mesh_path(name):
    return os.path.join(MESH_DIR, name)


@pytest.fixture(scope="session")
def bunny():
    return load_mesh(mesh_path("bunny.obj"))


@pytest.fixture(scope="session")
def sphere():
    return load_mesh(mesh_path("sphere.obj"))


@pytest.fixture(scope="session")
def cube():
    return load_mesh(mesh_path("cube.obj"))


@pytest.fixture(scope="session")
def bar():
    """Closed cylinder along z, length 1, radius 0.1 (GeomProc primitive)."""
    return from_geomproc(geomproc.create_cylinder(0.1, 1.0, 12, 21, closed=True))


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


def random_rotation(rng):
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def bar_bend(vertices, angle_degrees=90.0):
    """Fixed bottom of the bar; top rotated about x and moved onto an arc.

    Returns ``(constrained, targets, fixed, handles)``; ``constrained`` is
    sorted and ``targets`` follows it.
    """
    from arap import selection

    V = np.asarray(vertices)
    fixed = selection.by_coordinate(V, "z", below=0.05)
    handles = selection.by_coordinate(V, "z", above=0.95)
    z0, z1 = V[:, 2].min(), V[:, 2].max()
    length = z1 - z0
    theta = np.radians(angle_degrees)
    r = length / theta
    top = np.array([0.0, 0.0, z1])
    end = np.array([0.0, -r * (1 - np.cos(theta)), z0 + r * np.sin(theta)])
    R = selection.rotation("x", angle_degrees)
    targets = selection.rigid_targets(V[handles], R, pivot=top, translation=end - top)
    constrained, all_targets = selection.combine(V, fixed, handles, targets)
    return constrained, all_targets, fixed, handles
