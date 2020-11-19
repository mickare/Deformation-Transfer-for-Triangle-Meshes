from typing import Union, Sequence, Optional, Dict

import numpy as np
import plotly.graph_objects as go
import config
from meshlib import Mesh
from mathlib.vector import Vec3f

ODict = Optional[Dict]


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

    def addMesh(self, mesh: Mesh, offset: Optional[Vec3f] = None, **kwargs) -> "BrowserVisualizer":
        x, y, z = mesh.vertices.T[:3]
        if offset:
            x += offset[0]
            y += offset[1]
            z += offset[2]
        vx, vy, vz = mesh.faces.T
        mkwargs = dict(self.mesh_kwargs)
        mkwargs.update(kwargs)
        self._data.append(go.Mesh3d(x=x, y=y, z=z, i=vx, j=vy, k=vz, **mkwargs))
        return self

    def addScatter(self, points: Union[np.ndarray, Sequence[Vec3f]], offset: Optional[Vec3f] = None,
                   **kwargs) -> "BrowserVisualizer":
        pts = np.asarray(points)
        if offset:
            pts += offset
        x, y, z = pts.T
        self._data.append(go.Scatter3d(x=x, y=y, z=z, **kwargs))
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
                zaxis_title='Z',
                camera=camera,
                dragmode='orbit'
            ),
            scene_camera=camera
        )
        fig.show()


def plot_example1():
    """Simple example plot showing both a cat and a dog"""
    cat_path = "models/lowpoly_cat/cat_reference.obj"
    dog_path = "models/lowpoly_dog/dog_reference.obj"

    cat = Mesh.from_file_obj(cat_path)
    dog = Mesh.from_file_obj(dog_path)

    vis = BrowserVisualizer()
    vis.addMesh(cat, offset=(0, 0, cat.size()[2] + dog.size()[2]))
    vis.addMesh(dog)
    vis.addScatter(
        dog.vertices,
        marker=dict(
            color='red',
            size=3
        )
    )
    vis.show()


def plot_example2():
    """Simple example plot showing both a cat and a dog"""
    cat_path = "models/lowpoly_cat/cat_reference.obj"
    dog_path = "models/lowpoly_dog/dog_reference.obj"

    cat = Mesh.from_file_obj(cat_path)
    dog = Mesh.from_file_obj(dog_path)

    cat_index_label = [f"index: {i}" for i, v in enumerate(cat.vertices)]
    vis = BrowserVisualizer()
    vis.addMesh(cat, hovertext=cat_index_label)
    vis.addScatter(
        cat.vertices,
        marker=dict(
            color='red',
            size=4
        ),
        hovertext=cat_index_label
    )
    vis.show()

    dog_index_label = [f"index: {i}" for i, v in enumerate(dog.vertices)]
    vis = BrowserVisualizer()
    vis.addMesh(dog, hovertext=dog_index_label)
    vis.addScatter(
        dog.vertices,
        marker=dict(
            color='red',
            size=4
        ),
        hovertext=dog_index_label
    )
    vis.show()


def get_markers():
    markers = []
    with open(config.markers, 'r') as f:
        for line in f:
            if line[0] == "#":
                continue
            m = line.split(' ')
            markers.append((int(m[0]), int(m[1])))
    return np.array(markers)


def plot_example_markers():
    """Check if markers are correct"""

    cat = Mesh.from_file_obj(config.source_reference)
    dog = Mesh.from_file_obj(config.target_reference)

    for m in get_markers():
        cat.vertices[m[0]][0] = dog.vertices[m[1]][0]
        cat.vertices[m[0]][1] = dog.vertices[m[1]][1]
        cat.vertices[m[0]][2] = dog.vertices[m[1]][2]

    vis = BrowserVisualizer()
    vis.addMesh(cat)
    vis.addScatter(
        cat.vertices,
        marker=dict(
            color='red',
            size=3
        )
    )
    vis.show()


if __name__ == "__main__":
    plot_example()
