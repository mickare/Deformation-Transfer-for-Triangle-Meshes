from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import pywavefront
from scipy.cluster import vq

from mathlib.vector import Vec3f, Vector3D


@dataclass
class Mesh:
    """
    First simple data structure holding only the vertices and faces in a numpy array

    @param vertices th positions of triangle corners (x,y,z)
    @param faces the triangles (Triple of vertices indices)
    """
    vertices: np.ndarray
    faces: np.ndarray

    @classmethod
    def from_pywavefront(cls, obj: pywavefront.Wavefront):
        """
        Load a mesh from a pywavefront object
        :param obj:
        :return:
        """
        assert obj.mesh_list
        return cls(
            vertices=np.array(obj.vertices),
            faces=np.array(obj.mesh_list[0].faces)
        )

    @classmethod
    def from_file_obj(cls, file: str, **kwargs):
        """
        Load a mesh from a .obj file
        :param file:
        :param kwargs:
        :return:
        """
        kwargs.setdefault("encoding", "UTF-8")
        return cls.from_pywavefront(pywavefront.Wavefront(file, collect_faces=True, **kwargs))

    def scale(self, factor: float):
        """
        Scale the mesh
        :param factor:
        :return:
        """
        self.vertices *= factor
        return self

    def box(self) -> Tuple[Vec3f, Vec3f]:
        """
        Get the bounding box
        :return:
        """
        return np.min(self.vertices, axis=0), np.max(self.vertices, axis=0)

    def size(self) -> Vec3f:
        """
        Get the size of the mesh
        :return:
        """
        a, b = self.box()
        return b - a

    def move(self, offset: Vec3f):
        """
        Move the mesh
        :param offset:
        :return:
        """
        self.vertices += offset


class MeshAdaption:
    def __init__(self, transform: np.ndarray):
        assert transform.shape == (4, 4)
        self._transform = transform

    def apply(self, mesh: Mesh):
        return Mesh(
            vertices=Vector3D.apply(mesh.vertices, self._transform),
            faces=np.array(mesh.faces)
        )

    def reverse(self, mesh: Mesh):
        return Mesh(
            vertices=Vector3D.apply(mesh.vertices, np.linalg.inv(self._transform)),
            faces=np.array(mesh.faces)
        )

    @classmethod
    def unify_destination(cls, src: Mesh, dst: Mesh, markers: List[Tuple[int, int]]):
        if not markers:
            return cls(np.identity(4))
        elif len(markers) == 1:
            si, di = markers[0]
            return cls(
                Vector3D.new_offset(src.vertices[si] - dst.vertices[di])
            )
        else:
            m = np.array(markers)
            srcm = src.vertices[m[:, 0]]
            dstm = dst.vertices[m[:, 1]]
            spl = np.random.choice(len(markers), size=int(len(markers) / 2), replace=False)
            rest = np.array([i for i in range(len(markers)) if i not in spl])

            delta_src = np.median(srcm[spl], axis=0) - np.median(srcm[rest], axis=0)
            delta_dst = np.median(dstm[spl], axis=0) - np.median(dstm[rest], axis=0)

            # A dst ~= src
            # A dst - src ~= 0
