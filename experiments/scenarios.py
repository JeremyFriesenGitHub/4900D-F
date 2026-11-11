"""Deformation scenarios used by the experiments.

Each builder returns a ``Scenario`` with the rest mesh, the fixed vertices,
the handle vertices and the handle targets. Regions are chosen with
coordinate slabs relative to the bounding box (see ``arap.selection``), so
they do not depend on the mesh scale.
"""

import os
from dataclasses import dataclass

import numpy as np

import geomproc
from arap import selection as sel
from arap.meshio import from_geomproc, load_mesh

MESH_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "meshes")


@dataclass
class Scenario:
    name: str
    vertices: np.ndarray
    faces: np.ndarray
    fixed: np.ndarray
    handles: np.ndarray
    handle_targets: np.ndarray
    view: tuple = (20, -60)
    up: str = "z"
    description: str = ""

    @property
    def constraints(self):
        """Sorted ``(indices, targets)`` for ``ARAPDeformer``."""
        return sel.combine(self.vertices, self.fixed, self.handles, self.handle_targets)


def bar(rings=41, sides=20):
    """Closed cylinder along z with radius 0.1 and length 1."""
    return from_geomproc(geomproc.create_cylinder(0.1, 1.0, sides, rings, closed=True))


def bar_bend(angle=90.0, rings=41, sides=20):
    """Bottom of the bar fixed, top turned by ``angle`` and placed on an arc.

    The handle is moved to where it would be if the bar were bent into a
    circular arc of the same length, so a perfect isometric bend exists.
    """
    V, F = bar(rings, sides)
    fixed = sel.by_coordinate(V, "z", below=0.05)
    handles = sel.by_coordinate(V, "z", above=0.95)
    z0, z1 = V[:, 2].min(), V[:, 2].max()
    theta = np.radians(angle)
    radius = (z1 - z0) / theta
    top = np.array([0.0, 0.0, z1])
    end = np.array([0.0, -radius * (1.0 - np.cos(theta)), z0 + radius * np.sin(theta)])
    targets = sel.rigid_targets(V[handles], sel.rotation("x", angle), pivot=top,
                                translation=end - top)
    return Scenario("bar_bend", V, F, fixed, handles, targets, view=(10, 0),
                    description=f"bar bent by {angle:g} degrees")


def bar_twist(angle=135.0, rings=41, sides=20):
    """Bottom of the bar fixed, top rotated about the bar's own axis."""
    V, F = bar(rings, sides)
    fixed = sel.by_coordinate(V, "z", below=0.05)
    handles = sel.by_coordinate(V, "z", above=0.95)
    targets = sel.rigid_targets(V[handles], sel.rotation("z", angle), pivot=np.zeros(3))
    return Scenario("bar_twist", V, F, fixed, handles, targets, view=(15, -60),
                    description=f"bar twisted by {angle:g} degrees")


def hinge(angle=60.0):
    """GeomProc's two_part_cylinder: bottom fixed, top block turned about the neck."""
    V, F = load_mesh(os.path.join(MESH_DIR, "two_part_cylinder.obj"))
    fixed = sel.by_coordinate(V, "z", below=0.1)
    handles = sel.by_coordinate(V, "z", above=0.9)
    targets = sel.rigid_targets(V[handles], sel.rotation("x", angle), pivot=np.zeros(3))
    return Scenario("hinge", V, F, fixed, handles, targets, view=(10, 0),
                    description=f"two-part cylinder turned {angle:g} degrees at the neck")


def bunny_ears(shift=(-0.45, -0.15, 0.0)):
    """Bunny base fixed, ears pulled forward and down."""
    V, F = load_mesh(os.path.join(MESH_DIR, "bunny.obj"))
    fixed = sel.by_coordinate(V, "y", below=0.15)
    handles = sel.by_coordinate(V, "y", above=0.8)
    targets = V[handles] + np.asarray(shift)
    return Scenario("bunny_ears", V, F, fixed, handles, targets, view=(10, 0), up="y",
                    description="bunny ears pulled forward")


def kid_arms(angle=100.0, hand_radius=0.1):
    """Kid's feet fixed, both hands swung up sideways about the shoulders.

    The arms hang along the body, so each arm is the slab at one end of the
    y range. The hand is the part of it within ``hand_radius`` of its lowest
    point, and the shoulder pivot is the arm's top, moved slightly inwards.
    """
    V, F = load_mesh(os.path.join(MESH_DIR, "kid.obj"))
    fixed = sel.by_coordinate(V, "z", below=0.08)
    handles, targets = [], []
    for side, slab, sign in (("left", dict(below=0.1), -1.0), ("right", dict(above=0.9), 1.0)):
        arm = sel.by_coordinate(V, "y", **slab)
        z = V[arm, 2]
        hand_tip = V[arm[np.argmin(z)]]
        hand = arm[np.linalg.norm(V[arm] - hand_tip, axis=1) <= hand_radius]
        shoulder = V[arm[z > z.max() - 0.05]].mean(axis=0)
        shoulder[1] -= sign * 0.05
        handles.append(hand)
        targets.append(sel.rigid_targets(V[hand], sel.rotation("x", sign * angle), pivot=shoulder))
    return Scenario("kid_arms", V, F, fixed, np.concatenate(handles), np.vstack(targets),
                    view=(5, 0), description="kid raising both arms")


def camel_drink(pitch=-45.0, drop=0.55, forward=0.15):
    """Camel's hooves fixed, head lowered and tilted down as if drinking."""
    V, F = load_mesh(os.path.join(MESH_DIR, "camel.obj"))
    fixed = sel.by_coordinate(V, "y", below=0.1)
    handles = sel.by_coordinate(V, "z", above=0.9)
    pivot = V[handles].mean(axis=0)
    targets = sel.rigid_targets(V[handles], sel.rotation("x", -pitch), pivot=pivot,
                                translation=[0.0, -drop, forward])
    return Scenario("camel_drink", V, F, fixed, handles, targets, view=(10, -90), up="y",
                    description="camel lowering its head")


ALL = {
    "bar_bend": bar_bend,
    "bar_twist": bar_twist,
    "hinge": hinge,
    "bunny_ears": bunny_ears,
    "kid_arms": kid_arms,
    "camel_drink": camel_drink,
}
