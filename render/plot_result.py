from typing import Optional

import numpy as np
from plotly.graph_objs import Figure
from plotly.subplots import make_subplots

from config import ConfigFile
from meshlib import Mesh
from render.plot import BrowserVisualizer


def plot(source: Mesh, target: Mesh, vertices:bool =False, markers: Optional[np.ndarray] = None) -> Figure:
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        horizontal_spacing=0,
    )

    camera = dict(
        up=dict(x=0, y=1, z=0),
        eye=dict(x=1.7, y=2.0, z=1.8)
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
    hovertemplate = """
    <b>x:</b> %{x}<br>
    <b>y:</b> %{z}<br>
    <b>z:</b> %{y}<br>
    %{text}
    """
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
        ),
        hovertemplate=hovertemplate,
    )

    colorwheel = [
        'rgb(129,167, 62)', 'rgb( 57, 75, 124)', 'rgb(170,134, 57)', 'rgb(150, 57, 104)',
        'rgb( 48,123, 98)', 'rgb( 79, 44, 115)', 'rgb(170,193, 57)', 'rgb(180, 95, 67)'
    ]

    def getColor(n: int):
        return colorwheel[n % len(colorwheel)]

    # colorscale_rgb = [(0.0, 'rgb(255, 0, 0)'), (0.5, 'rgb(0, 255, 0)'), (1.0, 'rgb(0, 0, 255)')]

    source_rotated = source.transpose((0, 2, 1))
    target_rotated = target.transpose((0, 2, 1))

    # Plot markers
    if markers is not None:
        assert markers.shape[1] == 2
        fig.add_trace(
            BrowserVisualizer.make_scatter(
                source_rotated.vertices[markers[:, 0]],
                mode="text+markers",
                marker=dict(
                    color=[getColor(n) for n in range(len(markers))],
                    size=3,
                    symbol='x'
                ),
                hovertemplate=hovertemplate,
                text=[f"<b>Index:</b> {s}" for s, t in markers],
                name="Marker Source",
            ),
            row=1,
            col=1
        )
        fig.add_trace(
            BrowserVisualizer.make_scatter(
                target_rotated.vertices[markers[:, 1]],
                mode="text+markers",
                marker=dict(
                    color=[getColor(n) for n in range(len(markers))],
                    size=3,
                    symbol='x'
                ),
                hovertemplate=hovertemplate,
                text=[f"<b>Index:</b> {t}" for s, t in markers],
                name="Marker Target",
            ),
            row=1,
            col=2
        )

    # Plot Vertices
    if vertices:
        fig.add_trace(
            BrowserVisualizer.make_scatter(
                source_rotated.vertices,
                mode="markers",
                marker=dict(color='red', size=1, ),
                hovertemplate=hovertemplate,
                text=[f"<b>Vertex:</b> {n}" for n in range(len(source_rotated.vertices))],
                name="Vertex Source",
            ),
            row=1,
            col=1
        )
        fig.add_trace(
            BrowserVisualizer.make_scatter(
                target_rotated.vertices,
                mode="markers",
                marker=dict(color='blue', size=1, ),
                hovertemplate=hovertemplate,
                text=[f"<b>Vertex:</b> {n}" for n in range(len(target_rotated.vertices))],
                name="Vertex Target",
            ),
            row=1,
            col=2
        )

    # Plot meshes
    fig.add_trace(
        BrowserVisualizer.make_mesh(
            source_rotated,
            text=[f"<b>Vertex:</b> {n}" for n in range(len(source_rotated.vertices))],
            name="Source",
            **mesh_kwargs,
        ),
        row=1,
        col=1
    )
    fig.add_trace(
        BrowserVisualizer.make_mesh(
            target_rotated,
            text=[f"<b>Vertex:</b> {n}" for n in range(len(target_rotated.vertices))],
            name="Target",
            **mesh_kwargs,
        ),
        row=1,
        col=2
    )

    return fig


if __name__ == "__main__":
    cfg = ConfigFile.load(ConfigFile.Paths.highpoly.horse_camel)
    source = Mesh.load(cfg.source.reference)
    target = Mesh.load(cfg.target.reference)
    markers = cfg.markers

    plot(source, target).show(renderer="browser")
