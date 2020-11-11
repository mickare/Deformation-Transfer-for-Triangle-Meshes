from dataclasses import dataclass
from typing import Union, Dict, Optional, Tuple, Sequence

import numpy as np
import plotly
import pywavefront
import plotly.graph_objects as go

ODict = Optional[Dict]
Vec3f = Union[np.ndarray, Sequence[float], Tuple[float, float, float]]


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


class BrowserVisualizer:
    def __init__(self,
                 mesh_kwargs: ODict = None
                 ) -> None:
        self._data = []
        self.mesh_kwargs = mesh_kwargs or {}
        self.mesh_kwargs.setdefault("color", "gray")
        self.mesh_kwargs.setdefault("flatshading", True)
        self.mesh_kwargs["lighting"] = dict(
            ambient=0.1,
            diffuse=1.0,
            facenormalsepsilon=0.00000001,
            roughness=0.5,
            specular=0.4,
            fresnel=0.001
        )
        self.mesh_kwargs["lightposition"] = dict(
            x=-10000,
            y=10000,
            z=5000
        )

    def addMesh(self, mesh: Mesh) -> "BrowserVisualizer":
        x, y, z = mesh.vertices.T
        vx, vy, vz = mesh.faces.T
        self._data.append(go.Mesh3d(x=x, y=y, z=z, i=vx, j=vy, k=vz, **self.mesh_kwargs))
        return self

    def show(self, camera: ODict = None) -> None:
        camera = camera or {}
        camera.setdefault("up", dict(x=0, y=1, z=0))

        fig = go.Figure(
            data=self._data,
            layout=go.Layout(yaxis=dict(scaleanchor="x", scaleratio=1))
        )

        fig.update_layout(
            scene=dict(
                aspectmode='data',
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            scene_camera=camera
        )
        fig.show()


def example_plot():
    """Simple example plot showing both a cat and a dog"""
    cat_path = "models/lowpoly_cat/cat_reference.obj"
    dog_path = "models/lowpoly_dog/dog_reference.obj"

    cat = Mesh.from_file_obj(cat_path).scale(10)
    dog = Mesh.from_file_obj(dog_path)

    cat.move((0, 0, cat.size()[2] + dog.size()[2]))

    BrowserVisualizer().addMesh(cat).addMesh(dog).show()


if __name__ == "__main__":
    example_plot()
