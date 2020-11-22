import hashlib
import itertools
import math
from collections import defaultdict
from typing import List, Optional, Tuple, Dict, Set

import numpy as np
import tqdm
from scipy import sparse
from scipy.sparse.linalg import lsqr
from scipy.spatial import KDTree

import config
import meshlib
from render import BrowserVisualizer
from utils import SparseMatrixCache

original_source = meshlib.Mesh.from_file_obj(config.source_reference)
original_target = meshlib.Mesh.from_file_obj(config.target_reference)
markers = config.markers  # cat, dog
# markers = np.transpose((markers[:, 0], markers[:, 0]))

target_mesh = original_target.to_fourth_dimension()
subject = original_source.to_fourth_dimension()
# Show the source and target
# MeshPlots.side_by_side([original_source, original_target]).show(renderer="browser")

# Weights of cost functions
Ws = 1.0
Wi = 0.001
Wc = [1.0, 200.0, 1000.0, 5000.0]

# Precalculate the adjacent triangles in source
print("Prepare adjacent list")


def is_adjacent_edge(a: np.ndarray, b: np.ndarray):
    return any(
        (a[list(perm)] == b).sum() == 2 for perm in itertools.permutations((0, 1, 2), 3)
    )


def compute_adjacent_by_edges(mesh: meshlib.Mesh):
    """Computes the adjacent triangles by using the edges"""
    candidates = defaultdict(set)  # Edge -> Faces
    for n, f in enumerate(mesh.faces):
        f0, f1, f2 = sorted(f)
        candidates[(f0, f1)].add(n)
        candidates[(f0, f2)].add(n)
        candidates[(f1, f2)].add(n)

    faces_adjacent: Dict[int, Set[int]] = defaultdict(set)  # Face -> Faces
    for faces in candidates.values():
        for f in faces:
            faces_adjacent[f].update(faces)

    faces_sorted = sorted([(f, [a for a in adj if a != f]) for f, adj in faces_adjacent.items()], key=lambda e: e[0])
    return [adj for f, adj in faces_sorted]


def compute_adjacent_by_vertices(mesh: meshlib.Mesh):
    candidates = defaultdict(set)  # Vertex -> Faces
    for n, f in enumerate(mesh.faces):
        f0, f1, f2 = f
        candidates[f0].add(n)
        candidates[f1].add(n)
        candidates[f2].add(n)

    faces_adjacent: Dict[int, Set[int]] = defaultdict(set)  # Face -> Faces
    for faces in candidates.values():
        for f in faces:
            faces_adjacent[f].update(faces)

    faces_sorted = sorted([(f, [a for a in adj if a != f]) for f, adj in faces_adjacent.items()], key=lambda e: e[0])
    return [adj for f, adj in faces_sorted]


# adjacent = compute_adjacent_by_vertices(original_source)
adjacent = compute_adjacent_by_edges(original_source)


def get_closest_points(kd_tree: KDTree, verts: np.array):
    return kd_tree.query(verts)[1]


def get_aec(num_verts):
    return sparse.identity(num_verts * 3, dtype=np.float, format="csc")


def get_bec(closest_points: np.array, verts: np.array):
    return verts[closest_points]


class TransformEntry:
    """
    Class for creating the transformation matrix solution for T=xV^-1
    """

    def __init__(self, face: np.ndarray, invV: np.ndarray):
        assert face.shape == (4,)
        assert invV.shape == (3, 3)
        self.face = face
        """
        Solving
        x = [v2-v1, v3-v1, v4-v1]^-1
        w = xV^{-1}
        w_ij = v1 x2 - v1 x1 + v2 x3 - v2 x1 + v3 x4 - v3 x1
        w_ij = -(v1+v2+v3) x1 + (v1) x2 + (v2) x3 + (v3) x4
        """
        self.kleinA = np.zeros(shape=(9, 12))
        # Build T = V~ V^-1
        for i in range(3):  # Row of T
            for j in range(3):  # Column of T
                r = 3 * j + i
                self.kleinA[r, i] = - (invV[0, j] + invV[1, j] + invV[2, j])
                self.kleinA[r, i + 3] = invV[0, j]
                self.kleinA[r, i + 6] = invV[1, j]
                self.kleinA[r, i + 9] = invV[2, j]

    def insert_to(self, target: sparse.spmatrix, row: int, factor=1.0):
        # Index
        i0 = self.face[0] * 3
        i1 = self.face[1] * 3
        i2 = self.face[2] * 3
        i3 = self.face[3] * 3
        # Insert by adding
        part = self.kleinA * factor
        target[row:row + 9, i0:i0 + 3] += part[:, 0:3]
        target[row:row + 9, i1:i1 + 3] += part[:, 3:6]
        target[row:row + 9, i2:i2 + 3] += part[:, 6:9]
        target[row:row + 9, i3:i3 + 3] += part[:, 9:12]


#########################################################
print("Inverse Triangle Spans")
invVs = np.linalg.inv(subject.span)
assert len(subject.faces) == len(invVs)

#########################################################
# Preparing the transformation matrices
print("Preparing Transforms")
transforms = [TransformEntry(f, invV) for f, invV in zip(subject.faces, invVs)]


#########################################################
# Identity Cost - of transformations


def construct_identity_cost(subject, transforms) -> Tuple[sparse.spmatrix, np.ndarray]:
    AEi = sparse.dok_matrix(
        (
            # Count of all minimization terms
            len(subject.faces) * 9,
            # Length of flat result x
            len(subject.vertices) * 3
        ),
        dtype=np.float
    )
    for index, Ti in enumerate(tqdm.tqdm(transforms, desc="Building Identity Cost")):  # type: int, TransformEntry
        Ti.insert_to(AEi, row=index * 9)

    Bi = np.tile(np.identity(3, dtype=np.float).flatten(), len(subject.faces))

    assert AEi.shape[0] == len(Bi)
    return AEi.tocsr(), Bi


AEi, Bi = construct_identity_cost(subject, transforms)


#########################################################
# Smoothness Cost - of differences to adjacent transformations


def construct_smoothness_cost(subject, transforms, adjacent) -> Tuple[sparse.spmatrix, np.ndarray]:
    count_adjacent = sum(len(a) for a in adjacent)
    shape = (
        # Count of all minimization terms
        count_adjacent * 9,
        # Length of flat result x
        len(subject.vertices) * 3
    )

    hashid = hashlib.sha256()
    hashid.update(np.array(shape).data)
    hashid.update(subject.vertices.data)
    hashid = hashid.hexdigest()

    cache = SparseMatrixCache(suffix="_aes").entry(hashid=hashid, shape=shape)
    AEs = cache.get()

    if AEs is None:
        lhs = sparse.dok_matrix(shape, dtype=np.float)
        rhs = sparse.dok_matrix(shape, dtype=np.float)
        row = 0
        for index, Ti in enumerate(tqdm.tqdm(transforms, desc="Building Smoothness Cost")):
            for adj in adjacent[index]:
                Ti.insert_to(lhs, row)
                transforms[adj].insert_to(rhs, row)
                row += 9
        AEs = (lhs - rhs)
        assert row == AEs.shape[0]
        AEs = AEs.tocsr()
        cache.store(AEs)
    else:
        print("Reusing Smoothness Cost")

    Bs = np.zeros(count_adjacent * 9)
    assert AEs.shape[0] == len(Bs)
    return AEs, Bs


AEs, Bs = construct_smoothness_cost(subject, transforms, adjacent)

#
# def create_cost_smoothness() -> Tuple[spmatrix, np.ndarray]:
#     count_adjacent = sum(len(a) for a in adjacent)
#     Bs = np.zeros(count_adjacent * 9)
#     AEs = sparse.csc_matrix(
#         (
#             # Count of all minimization terms
#             count_adjacent * 9,
#             # Length of flat result x
#             len(subject.vertices) * 3
#         ),
#         dtype=np.float
#     )
#     assert AEs.shape[0] == len(Bs)
#     row = 0
#     for index, Ti in enumerate(tqdm.tqdm(transforms, desc="Building Smoothness Cost")):  # type: int, TransformEntry
#         for adj in adjacent[index]:
#             Ti.insert_to(AEs, row)
#             transforms[adj].insert_to(AEs, row, -1.0)
#             row += 9
#     assert row == AEs.shape[0]
#     return AEs, Bs


#########################################################
print("Building KDTree for closest points")
# KDTree for closest points in E_c
kd_tree_target = KDTree(target_mesh.vertices)
vertices: Optional[np.ndarray] = None

#########################################################
# Start of loop

total_steps = 6  # Steps per iteration
# Progress bar
pBar = tqdm.tqdm(total=iterations * total_steps)

for iteration in range(iterations):

    def pbar_next(msg: str):
        pBar.set_description(f"[{iteration + 1}/{iterations}] {msg}")
        pBar.update()


    Astack = [AEi * math.sqrt(Wi), AEs * math.sqrt(Ws)]
    Bstack = [Bi * math.sqrt(Wi), Bs * math.sqrt(Ws)]

    #########################################################
    pbar_next("Closest Point Costs")

    if iteration > 0:
        AEc = get_aec(len(subject.vertices))
        Bc = get_bec(get_closest_points(kd_tree_target, vertices), target_mesh.vertices).flatten()
        assert AEc.shape[0] == Bc.shape[0]
        Astack.append(AEc * math.sqrt(Wc[iteration]))
        Bstack.append(Bc * math.sqrt(Wc[iteration]))

    #########################################################
    pbar_next("Combining Costs")

    A = sparse.vstack(Astack, format="dok")
    b = np.concatenate(Bstack)

    #########################################################
    pbar_next("Enforcing Markers")
    for n, (mark_src_i, mark_dest_i) in enumerate(markers):
        i = mark_src_i * 3
        valueB = A[:, i:i + 3] @ target_mesh.vertices[mark_dest_i]
        b -= valueB
        A[:, i:i + 3] = 0

    #########################################################
    pbar_next("Solving")
    A = A.tocsc()

    # U, S, Vt = sparse.linalg.svds(A, k=len(original_source.vertices) * 3, which='LM')
    # result_verts = Vt.T @ sparse.diags([1.0 / S], [0]) @ U.T @ b

    x0 = subject.vertices
    assert A.shape[1] == x0.shape[0] * 3
    lsqr_result = lsqr(A, b, iter_lim=2000, show=True, x0=x0.flatten())
    result_verts = lsqr_result[0]

    #########################################################
    # Apply new vertices
    pbar_next("Applying vertices")
    vertices = result_verts[:len(subject.vertices) * 3].reshape((-1, 3))
    result = meshlib.Mesh(vertices=vertices[:len(original_source.vertices)],
                          faces=original_source.faces).to_fourth_dimension()
    # Enforce target vertices
    for mark_src_i, mark_dest_i in markers:
        result.vertices[mark_src_i] = target_mesh.vertices[mark_dest_i]

    #########################################################
    pbar_next("Rendering")

    vis = BrowserVisualizer()
    vis.add_mesh(result,
                 name=f"Result {iteration}",
                 text=[f"<b>Vertex:</b> {n}" for n in range(len(original_target.vertices))]
                 )
    vis.add_mesh(original_source,
                 name="Source",
                 color="red",
                 opacity=0.025,
                 # text=[f"<b>Vertex:</b> {n}" for n in range(len(original_target.vertices))]
                 hoverinfo='skip',
                 )
    vis.add_mesh(original_target,
                 name="Target",
                 color="blue",
                 opacity=0.025,
                 # text=[f"<b>Vertex:</b> {n}" for n in range(len(original_target.vertices))]
                 hoverinfo='skip',
                 )
    vis.add_scatter(
        original_target.vertices[markers[:, 1]],
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
        original_source.vertices[markers[:, 0]],
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
        original_target.vertices,
        marker=dict(
            color='blue',
            size=1,
            opacity=0.2,
        ),
        name="Vertex Target"
    )
    vis.show(renderer="browser")
