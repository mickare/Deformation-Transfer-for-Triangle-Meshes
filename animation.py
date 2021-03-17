"""
Transfers the animation from a model to the target.
"""

from typing import Sequence

import numpy as np
from plotly.graph_objs import Figure

import meshlib
from config import ConfigFile
from correspondence import get_correspondence
from render.plot import BrowserVisualizer
from transformation import Transformation
import plotly.graph_objects as go


def animate_cfg(cfg: ConfigFile, identity=False):
    corr_markers = cfg.markers  # List of vertex-tuples (source, target)
    if identity:
        corr_markers = np.ascontiguousarray(np.array((corr_markers[:, 0], corr_markers[:, 0]), dtype=np.int).T)

    original_source = meshlib.Mesh.load(cfg.source.reference)
    original_target = meshlib.Mesh.load(cfg.target.reference)
    if identity:
        original_target = meshlib.Mesh.load(cfg.source.reference)

    mapping = get_correspondence(original_source, original_target, corr_markers)
    transf = Transformation(original_source, original_target, mapping, smoothness=1)
    animate(transf, list(cfg.source.load_poses()))


def make_animation(transf: Transformation, poses: Sequence[meshlib.Mesh]):
    assert poses
    results = [transf(pose) for pose in poses]

    mesh_kwargs = dict(
        color='#ccc',
        opacity=1.0,
        flatshading=True,
        lighting=dict(
            ambient=0.1,
            diffuse=1.0,
            facenormalsepsilon=0.0000000000001,
            roughness=0.3,
            specular=0.7,
            fresnel=0.001
        ),
        lightposition=dict(
            x=-10000,
            y=10000,
            z=5000
        )
    )
    fig = Figure(
        data=[BrowserVisualizer.make_mesh(results[0].transpose((0, 2, 1)), **mesh_kwargs)],
        layout=dict(
            updatemenus=[
                dict(type="buttons",
                     buttons=[
                         dict(
                             label="Play",
                             method="animate",
                             args=[None, {
                                 "mode": "afterall",
                                 "frame": {"duration": 40, "redraw": True},
                                 "fromcurrent": False,
                                 "transition": {"duration": 40, "easing": "linear", "ordering": "traces first"}
                             }]
                         )
                     ])
            ],
        ),
        frames=[go.Frame(data=[BrowserVisualizer.make_mesh(p.transpose((0, 2, 1)), **mesh_kwargs)]) for p in results]
    )
    camera = dict(
        up=dict(x=0, y=1, z=0)
    )
    scene = dict(
        aspectmode='data',
        xaxis_title='X',
        yaxis_title='Z',
        zaxis_title='Y',
        camera=camera,
        dragmode='turntable'
    )
    fig.update_layout(
        scene=scene,
        scene2=scene,
        yaxis=dict(scaleanchor="x", scaleratio=1),
        yaxis2=dict(scaleanchor="x", scaleratio=1),
        margin=dict(l=0, r=0),
        # scene_camera=camera
    )
    return fig


def animate(transf: Transformation, poses: Sequence[meshlib.Mesh]):
    fig = make_animation(transf, poses)
    fig.show(renderer="browser")


if __name__ == "__main__":
    # cfg = ConfigFile.load(ConfigFile.Paths.highpoly.horse_camel)
    cfg = ConfigFile.load("models/lowpoly/markers-cat-voxel.yml")
    # cfg = ConfigFile.load(ConfigFile.Paths.highpoly.cat_lion)
    animate_cfg(cfg)
