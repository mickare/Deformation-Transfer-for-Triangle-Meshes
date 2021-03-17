from typing import Union, Sequence, Optional, Dict, List, Tuple, Any

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
            facenormalsepsilon=0.0000000000001,
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

    def finalize(self, camera: ODict = None) -> go.Figure:
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
        return fig

    def show(self, camera: ODict = None, **kwargs) -> None:
        fig = self.finalize(camera)
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

    @classmethod
    def plot_result_merged(cls, source: Mesh, target: Mesh, result: Mesh, markers: np.ndarray,
                           mesh_kwargs: Optional[Dict[str, Any]] = None):
        mesh_kwargs = mesh_kwargs or {}

        vis = BrowserVisualizer()
        vis.add_mesh(result,
                     name=f"Result",
                     text=[f"<b>Vertex:</b> {n}" for n in range(len(target.vertices))],
                     **mesh_kwargs
                     )
        vis.add_mesh(source,
                     name="Source",
                     color="red",
                     opacity=0.025,
                     # text=[f"<b>Vertex:</b> {n}" for n in range(len(target.vertices))]
                     hoverinfo='skip',
                     )
        vis.add_mesh(target,
                     name="Target",
                     color="blue",
                     opacity=0.025,
                     # text=[f"<b>Vertex:</b> {n}" for n in range(len(target.vertices))]
                     hoverinfo='skip',
                     )
        vis.add_scatter(
            target.vertices[markers[:, 1]],
            marker=dict(
                color='yellow',
                size=3,
                opacity=0.9,
                symbol='x',
            ),
            text=[f"<b>Index:</b> {t}" for s, t in markers],
            name="Marker Target"
        )
        vis.add_scatter(
            source.vertices[markers[:, 0]],
            marker=dict(
                color='red',
                size=3,
                opacity=0.9,
                symbol='x',
            ),
            text=[f"<b>Index:</b> {s}" for s, t in markers],
            name="Marker Source"
        )
        vis.add_scatter(
            target.vertices,
            marker=dict(
                color='blue',
                size=1,
                opacity=0.2,
            ),
            name="Vertex Target"
        )
        vis.show(renderer="browser")

    @classmethod
    def plot_correspondence(cls, source: Mesh, target: Mesh, correspondence: np.ndarray):
        assert correspondence.shape[1] == 2

        vis = BrowserVisualizer()
        vis.add_mesh(source,
                     name="Source",
                     color="red",
                     opacity=0.025,
                     # text=[f"<b>Vertex:</b> {n}" for n in range(len(target.vertices))]
                     hoverinfo='skip',
                     )
        vis.add_mesh(target,
                     name="Target",
                     color="blue",
                     opacity=0.025,
                     # text=[f"<b>Vertex:</b> {n}" for n in range(len(target.vertices))]
                     hoverinfo='skip',
                     )
        scent = source.get_centroids()
        tcent = target.get_centroids()

        scor = scent[correspondence.T[0]]
        tcor = tcent[correspondence.T[1]]
        lengths = np.linalg.norm(tcor - scor, axis=1)

        corres = np.array([e for s, t in zip(scor, tcor) for e in (s, t, (np.nan, np.nan, np.nan))][:-1])
        colors = np.array([c for l in lengths for c in (l, l, l)][:-1])

        vis.add_scatter(
            corres,
            marker=dict(
                line=dict(
                    color=colors,
                    colorscale="magma",
                    cauto=True
                ),
                # size=3,
                opacity=0.5,
                # symbol='x'
            ),
            mode="lines",
            name="Correspondence"
        )
        return vis


def plot_example2():
    """Simple example plot showing both a cat and a dog"""
    cat_path = "models/lowpoly/cat/cat_reference.obj"
    dog_path = "models/lowpoly/dog/dog_reference.obj"

    cat = Mesh.load(cat_path)
    dog = Mesh.load(dog_path)

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

    cat = Mesh.load(config.source_reference)
    dog = Mesh.load(config.target_reference)

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

    cat = Mesh.load(cat_path)
    dog = Mesh.load(dog_path)

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

def plot_voxel_cat():
    """Simple example plot showing both a cat and a dog"""
    cat_path = "models/lowpoly/cat_voxel/cat_voxel_mesh.npz"

    cat = Mesh.load(cat_path)

    vis = BrowserVisualizer()
    vis.add_mesh(cat)
    vis.show()

if __name__ == "__main__":
    plot_example1()
    # plot_voxel_cat()
    