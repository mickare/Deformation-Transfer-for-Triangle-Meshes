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

    @classmethod
    def make_mesh(cls, mesh: Mesh, offset: Optional[Vec3f] = None, heatmap=False, intensity=None, **kwargs):
        if heatmap and mesh.is_fourth_dimension():
            x, y, z = np.array(mesh.vertices.T)
            if offset is not None:
                x += offset[0]
                y += offset[1]
                z += offset[2]
            vx, vy, vz, vv4 = mesh.faces.T
            intensity = intensity or np.linalg.norm(mesh.vertices[vv4], axis=1)
            return go.Mesh3d(x=x, y=y, z=z, i=vx, j=vy, k=vz,
                             intensity=intensity,
                             intensitymode="cell",
                             **kwargs)
        else:
            mesh = mesh.to_third_dimension(copy=False)
            x, y, z = np.array(mesh.vertices.T)
            if offset is not None:
                x += offset[0]
                y += offset[1]
                z += offset[2]
            vx, vy, vz = mesh.faces.T
            return go.Mesh3d(x=x, y=y, z=z, i=vx, j=vy, k=vz, **kwargs)

    def add_mesh(self, mesh: Mesh, *args, **kwargs) -> "BrowserVisualizer":
        mkwargs = dict(self.mesh_kwargs)
        mkwargs.update(kwargs)
        self._data.append(self.make_mesh(mesh, *args, **mkwargs))
        return self

    @classmethod
    def make_scatter(cls, points: Union[np.ndarray, Sequence[Vec3f]], offset: Optional[Vec3f] = None, **kwargs):
        kwargs.setdefault("marker", {})
        kwargs["marker"].setdefault("size", 0.1)
        kwargs.setdefault("mode", "markers")

        pts = np.asarray(points)
        if offset is not None:
            pts += offset
        x, y, z = pts.T
        return go.Scatter3d(x=x, y=y, z=z, **kwargs)

    def add_scatter(self, points: Union[np.ndarray, Sequence[Vec3f]], offset: Optional[Vec3f] = None,
                    **kwargs) -> "BrowserVisualizer":
        self._data.append(self.make_scatter(points, offset, **kwargs))
        return self

    def show(self, camera: ODict = None, **kwargs) -> None:
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
        fig.show(**kwargs)


class MeshPlots:
    @classmethod
    def side_by_side(cls, meshes: Sequence[Mesh], spacing=0.5, axis=2) -> BrowserVisualizer:
        vis = BrowserVisualizer()
        offset = np.zeros(3)
        for m in meshes:
            size = m.size()[axis]
            vis.add_mesh(m, offset=offset)
            offset += (0, 0, size + spacing)
        return vis


def plot_example2():
    """Simple example plot showing both a cat and a dog"""
    cat_path = "models/lowpoly/cat/cat_reference.obj"
    dog_path = "models/lowpoly/dog/dog_reference.obj"

    cat = Mesh.from_file_obj(cat_path)
    dog = Mesh.from_file_obj(dog_path)

    cat_index_label = [f"index: {i}" for i, v in enumerate(cat.vertices)]
    vis = BrowserVisualizer()
    vis.add_mesh(cat, hovertext=cat_index_label)
    vis.add_scatter(
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
    vis.add_mesh(dog, hovertext=dog_index_label)
    vis.add_scatter(
        dog.vertices,
        marker=dict(
            color='red',
            size=4
        ),
        hovertext=dog_index_label
    )
    vis.show()


def plot_example_markers():
    """Check if markers are correct"""

    cat = Mesh.from_file_obj(config.source_reference)
    dog = Mesh.from_file_obj(config.target_reference)

    for m in config.markers:
        cat.vertices[m[0]][0] = dog.vertices[m[1]][0]
        cat.vertices[m[0]][1] = dog.vertices[m[1]][1]
        cat.vertices[m[0]][2] = dog.vertices[m[1]][2]

    vis = BrowserVisualizer()
    vis.add_mesh(cat)
    vis.add_scatter(
        cat.vertices,
        marker=dict(
            color=np.random.choice(len(cat.vertices), replace=False),
            colorscale='Viridis',
            size=2,
        )
    )
    vis.show()


def plot_example1():
    """Simple example plot showing both a cat and a dog"""
    cat_path = "models/lowpoly/cat/cat_reference.obj"
    dog_path = "models/lowpoly/dog/dog_reference.obj"

    cat = Mesh.from_file_obj(cat_path)
    dog = Mesh.from_file_obj(dog_path)

    vis = BrowserVisualizer()
    vis.add_mesh(cat, offset=(0, 0, cat.size()[2] + dog.size()[2]))
    vis.add_mesh(dog)
    vis.add_scatter(
        dog.vertices,
        marker=dict(
            # color=np.random.choice(len(dog.vertices), len(dog.vertices), replace=False),
            # color=np.sin(np.linalg.norm(dog.vertices, axis=1) * 100),
            color=np.sin(np.arange(len(dog.vertices))),
            colorscale='Viridis',
            size=2,
        )
    )
    vis.show()


if __name__ == "__main__":
    plot_example1()
