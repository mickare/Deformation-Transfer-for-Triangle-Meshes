from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

import meshlib
from mathlib.vector import Vector3D
from meshlib import Mesh
from render import BrowserVisualizer


@dataclass
class TriangleSpanMesh:
    vertices: np.ndarray
    faces: np.ndarray

    span: np.ndarray
    span_offset: np.ndarray

    # @classmethod
    # def calc_fourth_vector(cls, mesh: Mesh) -> np.ndarray:
    #     v1, v2, v3 = mesh.vertices[mesh.faces].transpose((1, 0, 2))
    #     a = v2 - v1
    #     b = v3 - v1
    #     tmp = np.cross(a, b)
    #     return v1 + (tmp.T / np.sqrt(np.linalg.norm(tmp, axis=1))).T

    @classmethod
    def from_mesh(cls, mesh: Mesh):
        return cls.from_parts(mesh.vertices, mesh.faces)

    @classmethod
    def from_parts(cls, vertices: np.ndarray, faces: np.ndarray) -> "TriangleSpanMesh":
        """
        Computes the vectors for the closed form equation in the paper (eq. 3).
        It skips the step of computing v4.
        :param vertices:
        :param faces:
        :return:
        """
        a, b, c = cls._calc_span(vertices, faces)
        v1 = vertices[faces][:, 0]
        v4 = v1 + c
        new_vertices = np.concatenate((vertices, v4), axis=0)
        v4_indices = np.arange(len(vertices), len(vertices) + len(c))
        new_faces = np.concatenate((faces, v4_indices.reshape((-1, 1))), axis=1)
        return cls(new_vertices, new_faces, np.transpose((a, b, c), axes=(1, 0, 2)), v1)

    @classmethod
    def _calc_span(cls, vertices: np.ndarray, faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        v1, v2, v3 = vertices[faces].transpose((1, 0, 2))
        a = v2 - v1
        b = v3 - v1
        tmp = np.cross(a, b)
        c = (tmp.T / np.sqrt(np.linalg.norm(tmp, axis=1))).T
        return a, b, c

    def update_span(self, vertices: Optional[np.ndarray] = None, faces: Optional[np.ndarray] = None):
        vertices = vertices or self.vertices
        faces = faces or self.faces
        if vertices.shape[1] == 4:
            vertices = vertices[:, :3]
        if faces.shape[1] == 4:
            faces = faces[:, :3]
        a, b, c = self._calc_span(vertices, faces)
        self.span = np.transpose((a, b, c), axes=(1, 0, 2))
        self.span_offset = vertices[:, 0]
        return a, b, c

    def to_mesh(self) -> meshlib.Mesh:
        return meshlib.Mesh(
            vertices=self.vertices[:, :3],
            faces=self.faces[:, :3]
        )


def example_discrete_mesh_rotation(file="models/lowpoly_cat/cat_reference.obj"):
    cat = meshlib.Mesh.from_file_obj(file)
    dmesh = TriangleSpanMesh.from_mesh(cat)

    rot = Vector3D.new_rotation((1, 0, 0), np.pi * 0.2)
    dmesh.span = np.array([Vector3D.apply(tri, rot) for tri in dmesh.span])

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
