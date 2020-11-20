from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

import meshlib
from mathlib.vector import Vector3D
from meshlib import Mesh
from render import BrowserVisualizer


@dataclass()
class TriangleSpanMesh:
    span: np.ndarray
    offset: np.ndarray

    @classmethod
    def from_mesh(cls, mesh: Mesh):
        return cls(mesh.span, mesh.v1)

    def to_mesh(self) -> meshlib.Mesh:
        vertices, faces = zip(*[((v1, v1 + d1, v1 + d2), (i * 3, i * 3 + 1, i * 3 + 2))
                                for i, (v1, (d1, d2, n3)) in enumerate(zip(self.offset, self.span))])
        return meshlib.Mesh(
            vertices=np.vstack(vertices),
            faces=np.array(faces, dtype=np.int)
        )


def example_discrete_mesh_rotation(file="models/lowpoly_cat/cat_reference.obj"):
    cat = meshlib.Mesh.from_file_obj(file)
    dmesh = TriangleSpanMesh.from_mesh(cat)

    rot = Vector3D.new_rotation((1, 0, 0), np.pi * 0.2)
    dmesh.span = np.array([Vector3D.apply(tri, rot) for tri in dmesh.span])

    vis = BrowserVisualizer()
    vis.addMesh(dmesh.to_mesh())
    vis.addScatter(
        cat.vertices,
        marker=dict(
            color='red',
            size=3
        )
    )
    vis.show()


if __name__ == "__main__":
    example_discrete_mesh_rotation()
