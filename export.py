"""
Exports all views to a html page.
"""
import os

import meshlib
import numpy as np

from animation import make_animation
from config import ConfigFile
from correspondence import get_correspondence
from transformation import Transformation

import render.plotly_html

if __name__ == "__main__":
    import render.plot_result as plt_res
    import render.plot as plt

    name = "horsecamel"
    cfg = ConfigFile.load(ConfigFile.Paths.highpoly.horse_camel)

    # name = "catdog"
    # cfg = ConfigFile.load(ConfigFile.Paths.lowpoly.catdog)

    # name = "catlion"
    # cfg = ConfigFile.load(ConfigFile.Paths.highpoly.cat_lion)

    # name = "catvoxel"
    # cfg = ConfigFile.load(ConfigFile.Paths.lowpoly.catvoxel)

    corr_markers = cfg.markers  # List of vertex-tuples (source, target)

    identity = False
    if identity:
        corr_markers = np.ascontiguousarray(
            np.array((corr_markers[:, 0], corr_markers[:, 0]), dtype=np.int).T
        )

    #########################################################
    # Load meshes

    original_source = meshlib.Mesh.load(cfg.source.reference)
    original_pose = meshlib.Mesh.load(cfg.source.poses[0])
    original_target = meshlib.Mesh.load(cfg.target.reference)
    if identity:
        original_target = meshlib.Mesh.load(cfg.source.reference)

    #########################################################
    # Load correspondence from cache if possible
    mapping = get_correspondence(
        original_source, original_target, corr_markers, plot=False
    )

    transf = Transformation(original_source, original_target, mapping)
    result = transf(original_pose)

    path = f"result/{name}"
    os.makedirs(path, exist_ok=True)
    plt.MeshPlots.plot_correspondence(
        original_source, original_target, mapping
    ).finalize().write_html(
        os.path.join(path, "correspondence.html"), include_plotlyjs="cdn"
    )
    plt_res.plot(original_source, original_target).write_html(
        os.path.join(path, "reference.html"), include_plotlyjs="cdn"
    )
    plt_res.plot(original_pose, result).write_html(
        os.path.join(path, "result.html"), include_plotlyjs="cdn"
    )

    poses_meshes = list(cfg.source.load_poses())
    make_animation(transf, poses_meshes).write_html(
        os.path.join(path, "animation.html"), include_plotlyjs="cdn"
    )

    poses = []
    for n, pose in enumerate(poses_meshes):
        filename = f"pose.{n}.html"
        poses.append(filename)
        plt_res.plot(pose, transf(pose)).write_html(
            os.path.join(path, filename), include_plotlyjs="cdn"
        )

    files = [
        "reference.html",
        "correspondence.html",
        "result.html",
        "animation.html",
        *poses,
    ]

    html = f"""<html>
        <head>
            <title>{name}</title>"""
    html += """
            <style>
            html, body {
              height: 100%;
              margin: 0;
            }
            #iframe_page {
                width:100%;height:100%;
                display: inline-block;
                padding:0; margin:0;
            }
            
            /* DivTable.com */
            .divTable{
                display: table;
                width: 100%; height:100%;
            }
            .divTableRow {
                display: table-row;
            }
            .divTableHeading {
                background-color: #EEE;
                display: table-header-group;
            }
            .divTableCell, .divTableHead {
                border: 1px solid #999999;
                display: table-cell;
                padding: 3px 10px;
            }
            .divTableHeading {
                background-color: #EEE;
                display: table-header-group;
                font-weight: bold;
            }
            .divTableFoot {
                background-color: #EEE;
                display: table-footer-group;
                font-weight: bold;
            }
            .divTableBody {
                display: table-row-group;
            }
            </style>
        </head>
        <body>
        <div class="divTable">
        <div class="divTableBody">
        <div class="divTableRow">
        <div class="divTableCell" style="vertical-align: top; max-width: 40px;">
            <div>
            <ul>
        """

    for file in files:
        html += f'<li><a href="{file}" target="page">{file}</a></li>'

    html += f"""
            </ul>
</div>
        </div>
        <div class="divTableCell" style="position:relative;padding:0; margin:0;"><iframe src="{files[0]}" name="page" id="iframe_page"></iframe></div>
        </div>
        </div>
        </div>
        </body>
        </html>
            """

    with open(os.path.join(path, "index.html"), "wt") as f:
        f.writelines(html)
