from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pywavefront

from mathlib.vector import Vec3f


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