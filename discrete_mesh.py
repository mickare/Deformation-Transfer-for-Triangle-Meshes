from dataclasses import dataclass

import numpy as np

import meshlib
from mathlib.vector import Vector3D
from meshlib import Mesh
from render import BrowserVisualizer


@dataclass
class DiscreteMesh:
    span_vertices: np.ndarray
    offset: np.ndarray

    @classmethod
    def calc_triangle_norm(cls, mesh: Mesh) -> np.ndarray:
        v1, v2, v3 = mesh.vertices[mesh.faces].transpose((1, 0, 2))
        a = v2 - v1
        b = v3 - v1
        tmp = np.cross(a, b)
        return v1 + tmp / np.sqrt(np.linalg.norm(tmp))

    @classmethod
    def from_mesh(cls, mesh: Mesh) -> "DiscreteMesh":
        """
        Computes the vectors for the closed form equation in the paper (eq. 3).
        It skips the step of computing v4.
        :param vertices:
        :param faces:
        :return:
        """
        v1, v2, v3 = mesh.vertices[mesh.faces].transpose((1, 0, 2))
        # v1 = mesh.vertices[mesh.faces[:, 0]]
        # v2 = mesh.vertices[mesh.faces[:, 1]]
        # v3 = mesh.vertices[mesh.faces[:, 2]]
        a = v2 - v1
        b = v3 - v1
        tmp = np.cross(a, b)
        c = tmp / np.sqrt(np.linalg.norm(tmp))
        return cls(np.transpose((a, b, c), axes=(1, 0, 2)), v1)

    def to_mesh_simple(self) -> meshlib.Mesh:
        vertices, faces = zip(*[((v1, v1 + d1, v1 + d2), (i * 3, i * 3 + 1, i * 3 + 2))
                                for i, (v1, (d1, d2, n3)) in enumerate(zip(self.offset, self.span_vertices))])
        return meshlib.Mesh(np.concatenate(vertices, axis=0), np.array(faces))


def example_discrete_mesh_rotation(file="models/lowpoly_cat/cat_reference.obj"):
    cat = meshlib.Mesh.from_file_obj(file)
    dmesh = DiscreteMesh.from_mesh(cat)

    rot = Vector3D.new_rotation((1, 0, 0), np.pi * 0.2)
    dmesh.span_vertices = np.array([Vector3D.apply(tri, rot) for tri in dmesh.span_vertices])

    vis = BrowserVisualizer()
    vis.addMesh(dmesh.to_mesh_simple())
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
