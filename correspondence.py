"""
Computes the correspondence between vertices of two models.
It "inflates" the source mesh until it fits the target mesh (by minimizing a cost function).

This implementation is an approximation of the paper solution, since it simplifies the problem
by matching the source vertices to the target vertices.
But a better solution would be to match the source vertice to the target surfaces.
"""


import hashlib
from collections import defaultdict
from typing import Tuple, Dict, Set, List, Optional

import numpy as np
import tqdm
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_linalg
from scipy.spatial import cKDTree

import meshlib
from config import ConfigFile
from render.plot import MeshPlots
from meshlib.cache import SparseMatrixCache, CorrespondenceCache


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


#########################################################
# Closest point search

def get_aec(columns, rows):
    return sparse.identity(columns, dtype=float, format="csc")[:rows]


def get_bec(closest_points: np.array, verts: np.array):
    return verts[closest_points]


# def fallback_closest_points(kd_tree: cKDTree, vert: np.ndarray, normal: np.ndarray, target_normals: np.ndarray,
#                             max_angle: float = np.radians(90)) -> int:
#     for i in range(int(np.ceil(len(target_normals) / 1000))):
#         start = i * 1000 + 1
#         ks = np.arange(start, min(start + 1000, len(target_normals)))
#         dist, ind = kd_tree.query(vert, ks)
#         angles = np.arccos(np.dot(target_normals[ind], normal))
#         angles_cond = np.abs(angles) < max_angle
#         if angles_cond.any():
#             return ind[angles_cond][0]
#     raise RuntimeError("Could not find any point on the target mesh!")


def get_closest_points(kd_tree: cKDTree, verts: np.array, vert_normals: np.array, target_normals: np.array,
                       max_angle: float = np.radians(90), ks=200) -> np.ndarray:
    assert len(verts) == len(vert_normals)
    closest_points: List[Tuple[int, int]] = []
    dists, indicies = kd_tree.query(verts, min(len(target_normals), ks))
    for v, (dist, ind) in enumerate(zip(dists, indicies)):
        angles = np.arccos(np.dot(target_normals[ind], vert_normals[v]))
        angles_cond = np.abs(angles) < max_angle
        if angles_cond.any():
            cind = ind[angles_cond][0]
            closest_points.append((v, cind))
        else:
            # Fallback
            # cind = fallback_closest_points(kd_tree, verts[v], vert_normals[v], target_normals, max_angle)
            # closest_points.append(cind)
            pass

    return np.array(closest_points)


def get_vertex_normals(verts: np.array, faces: np.array) -> np.ndarray:
    max_index = np.max(faces[:, :3]) + 1
    candidates = [set() for i in range(max_index)]
    for n, (f0, f1, f2) in enumerate(faces[:, :3]):
        candidates[f0].add(n)
        candidates[f1].add(n)
        candidates[f2].add(n)

    """
    Normals only point in the correct direction if the vertices are in the right order in faces. This might not hold for 
    meshes created by scanners.
    """
    triangle_normals = get_triangle_normals(verts, faces)
    triangle_normals_per_vertex = [[triangle_normals[i] for i in indices] for indices in candidates]
    vertex_normals = np.array([
        np.mean(normals, 0) if normals else np.zeros(3, float) for normals in triangle_normals_per_vertex
    ])
    assert len(vertex_normals) == max_index
    return (vertex_normals.T / np.linalg.norm(vertex_normals, axis=1)).T


def get_triangle_normals(verts: np.array, faces: np.array):
    vns = np.cross(verts[faces[:, 1]] - verts[faces[:, 0]], verts[faces[:, 2]] - verts[faces[:, 0]])
    return (vns.T / np.linalg.norm(vns, axis=1)).T


def max_triangle_length(mesh: meshlib.Mesh):
    a, b, c = mesh.span_components()
    return max(np.max(np.linalg.norm(a, axis=1)), np.max(np.linalg.norm(b, axis=1)))


def match_triangles(source: meshlib.Mesh, target: meshlib.Mesh, factor=2) -> List[Tuple[int, int]]:
    source_centroids = source.get_centroids()
    target_centroids = target.get_centroids()
    source_normals = source.normals()
    target_normals = target.normals()
    radius = max(max_triangle_length(source), max_triangle_length(target)) * factor
    triangles = get_closest_triangles(source_normals, target_normals, source_centroids, target_centroids, radius)
    tmp_triangles = get_closest_triangles(target_normals, source_normals, target_centroids, source_centroids,
                                          radius)
    triangles.update((t[1], t[0]) for t in tmp_triangles)
    return list(triangles)


def get_closest_triangles(
        source_normals: np.ndarray,
        target_normals: np.ndarray,
        source_centroids: np.ndarray,
        target_centroids: np.ndarray,
        max_angle: float = np.radians(90),
        k: int = 500,
        radius: float = np.inf
) -> Set[Tuple[int, int]]:
    assert len(source_normals) == len(source_centroids)
    assert len(target_normals) == len(target_centroids)
    triangles = set()
    kd_tree = cKDTree(target_centroids)

    dists, indicies = kd_tree.query(source_centroids, min(len(target_centroids), k), distance_upper_bound=radius)
    for index_source, (dist, ind) in enumerate(zip(dists, indicies)):
        angles = np.arccos(np.dot(target_normals[ind], source_normals[index_source]))
        angles_cond = angles < max_angle
        if angles_cond.any():
            index_target = ind[angles_cond][0]
            triangles.add((index_source, index_target))

    return triangles


#########################################################
# Matrix builder for T Transformation entries

class TransformMatrix:
    __row_partial_baked = np.array([0, 1, 2] * 4)

    @classmethod
    def expand(cls, f: np.ndarray, inv: np.ndarray, size: int):
        i0, i1, i2, i3 = f
        col = np.array([i0, i0, i0, i1, i1, i1, i2, i2, i2, i3, i3, i3])
        data = np.concatenate([-inv.sum(axis=0), *inv])
        return sparse.coo_matrix((data, (cls.__row_partial_baked, col)), shape=(3, size), dtype=float)

    @classmethod
    def construct(cls, faces: np.ndarray, invVs: np.ndarray, size: int, desc="Building Transformation Matrix"):
        assert len(faces) == len(invVs)
        return sparse.vstack([
            cls.expand(f, inv, size) for f, inv in tqdm.tqdm(zip(faces, invVs), total=len(faces), desc=desc)
        ], dtype=float)


def apply_markers(A: sparse.spmatrix, b: np.ndarray, target: meshlib.Mesh, markers: np.ndarray) \
        -> Tuple[sparse.spmatrix, np.ndarray]:
    """
    Solves the marker vertices of `target` in `A` and pushes it to the right side of the equation `Ax=b` into `b`.
    Returns a new matrix of `A` without the columns of the markers and the new result vector `b'`.
    :param A: Matrix (NxM)
    :param b: Result vector (Nx3)
    :param target: Target mesh
    :param markers: Marker (Qx2) with first column the source indices and the second the target indices.
    :return: Matrix (Nx(M-Q)), result vector (Nx3)
    """
    assert markers.ndim == 2 and markers.shape[1] == 2
    invmarker = np.setdiff1d(np.arange(A.shape[1]), markers[:, 0])
    zb = b - A[:, markers.T[0]] * target.vertices[markers.T[1]]
    return A[:, invmarker].tocsc(), zb


def revert_markers(A: sparse.spmatrix, x: np.ndarray, target: meshlib.Mesh, markers: np.ndarray,
                   *, out: Optional[np.ndarray] = None):
    if out is None:
        out = np.zeros((A.shape[1] + len(markers), 3))
    else:
        assert out.shape == (A.shape[1] + len(markers), 3)
    invmarker = np.setdiff1d(np.arange(len(out)), markers[:, 0])
    out[invmarker] = x
    out[markers[:, 0]] = target.vertices[markers[:, 1]]
    return out


#########################################################
# Identity Cost - of transformations


def construct_identity_cost(subject, invVs) -> Tuple[sparse.spmatrix, np.ndarray]:
    """ Construct the terms for the identity cost """
    shape = (
        # Count of all minimization terms
        len(subject.faces) * 3,
        # Length of flat result x
        len(subject.vertices)
    )

    hashid = hashlib.sha256()
    hashid.update(b"identity")
    hashid.update(np.array(shape).data)
    hashid.update(subject.vertices.data)
    hashid = hashid.hexdigest()

    cache = SparseMatrixCache(suffix="_aei").entry(hashid=hashid, shape=shape)
    AEi = cache.get()

    if AEi is None:
        AEi = TransformMatrix.construct(
            subject.faces, invVs, len(subject.vertices),
            desc="Building Identity Cost"
        ).tocsr()
        AEi.eliminate_zeros()
        cache.store(AEi)
    else:
        print("Reusing Identity Cost")

    Bi = np.tile(np.identity(3, dtype=float), (len(subject.faces), 1))
    assert AEi.shape[0] == Bi.shape[0]
    return AEi.tocsr(), Bi


#########################################################
# Smoothness Cost - of differences to adjacent transformations


def construct_smoothness_cost(subject, invVs, adjacent) -> Tuple[sparse.spmatrix, np.ndarray]:
    """ Construct the terms for the Smoothness cost"""
    count_adjacent = sum(len(a) for a in adjacent)
    shape = (
        # Count of all minimization terms
        count_adjacent * 3,
        # Length of flat result x
        len(subject.vertices)
    )

    hashid = hashlib.sha256()
    hashid.update(b"smoothness")
    hashid.update(np.array(shape).data)
    hashid.update(subject.vertices.data)
    hashid = hashid.hexdigest()

    cache = SparseMatrixCache(suffix="_aes").entry(hashid=hashid, shape=shape)
    AEs = cache.get()

    if AEs is None:
        size = len(subject.vertices)

        def construct(f, inv, index):
            a = TransformMatrix.expand(f, inv, size).tocsc()
            for adj in adjacent[index]:
                yield a, TransformMatrix.expand(subject.faces[adj], invVs[adj], size).tocsc()

        lhs, rhs = zip(*(adjacents for index, (f, inv) in
                         enumerate(tqdm.tqdm(zip(subject.faces, invVs), total=len(subject.faces),
                                             desc="Building Smoothness Cost"))
                         for adjacents in construct(f, inv, index)))
        AEs = sparse.vstack(lhs) - sparse.vstack(rhs)

        # AEs = sparse.vstack([
        #     adjacents for index, (f, inv) in
        #     enumerate(tqdm.tqdm(zip(subject.faces, invVs), total=len(subject.faces), desc="Building Smoothness Cost"))
        #     for adjacents in construct(f, inv, index)
        # ], dtype=float).tocsr()
        AEs.eliminate_zeros()
        cache.store(AEs)
    else:
        print("Reusing Smoothness Cost")

    Bs = np.zeros((count_adjacent * 3, 3))
    assert AEs.shape[0] == Bs.shape[0]
    return AEs, Bs


def compute_correspondence(source_org: meshlib.Mesh, target_org: meshlib.Mesh, markers: np.ndarray, plot=False) \
        -> np.ndarray:
    #########################################################
    # Configuration

    # Meshes
    # cfg = ConfigFile.load(ConfigFile.Paths.lowpoly.catdog)
    # cfg = ConfigFile.load(ConfigFile.Paths.highpoly.horse_camel)

    # Weights of cost functions
    Ws = 1.0
    Wi = 0.001
    Wc = [0, 10, 50, 250, 1000, 2000, 3000, 5000]

    #########################################################

    source = source_org.to_fourth_dimension()
    target = target_org.to_fourth_dimension()
    # Show the source and target
    # MeshPlots.side_by_side([original_source, original_target]).show(renderer="browser")

    #########################################################
    # Precalculate the adjacent triangles in source
    print("Precalculate adjacent list")

    # adjacent = compute_adjacent_by_vertices(source_org)
    adjacent = compute_adjacent_by_edges(source_org)

    #########################################################
    print("Inverse Triangle Spans")
    invVs = np.linalg.inv(source.span)
    assert len(source.faces) == len(invVs)

    #########################################################
    # Preparing the transformation matrices
    print("Preparing Transforms")
    # transforms = [TransformEntry(f, invV) for f, invV in zip(source.faces, invVs)]

    AEi, Bi = apply_markers(*construct_identity_cost(source, invVs), target, markers)

    AEs, Bs = apply_markers(*construct_smoothness_cost(source, invVs, adjacent), target, markers)

    #########################################################
    print("Building KDTree for closest points")
    # KDTree for closest points in E_c
    kd_tree_target = cKDTree(target_org.vertices)
    target_normals = get_vertex_normals(target_org.vertices, target_org.faces)
    vertices: np.ndarray = np.copy(source.vertices)

    #########################################################
    # Start of loop

    iterations = len(Wc)
    total_steps = 3  # Steps per iteration
    if plot:
        total_steps += 1

    # Progress bar
    pBar = tqdm.tqdm(total=iterations * total_steps)

    for iteration in range(iterations):

        def pbar_next(msg: str):
            pBar.set_description(f"[{iteration + 1}/{iterations}] {msg}")
            pBar.update()

        Astack = [AEi * Wi, AEs * Ws]
        Bstack = [Bi * Wi, Bs * Ws]

        #########################################################
        pbar_next("Closest Point Costs")

        if iteration > 0 and Wc[iteration] != 0:
            AEc = get_aec(len(source.vertices), len(source_org.vertices))
            vertices_clipped = vertices[:len(source_org.vertices)]
            closest_points = get_closest_points(kd_tree_target, vertices_clipped,
                                                get_vertex_normals(vertices_clipped, source_org.faces), target_normals)
            AEc = AEc[closest_points[:, 0]]
            Bc = get_bec(closest_points[:, 1], target.vertices)
            assert AEc.shape[0] == Bc.shape[0]

            mAEc, mBc = apply_markers(AEc, Bc, target, markers)
            Astack.append(mAEc * Wc[iteration])
            Bstack.append(mBc * Wc[iteration])

        #########################################################
        pbar_next("Combining Costs")

        A: sparse.spmatrix = sparse.vstack(Astack, format="csc")
        A.eliminate_zeros()
        b = np.concatenate(Bstack)

        #########################################################
        pbar_next("Solving")
        A = A.tocsc()

        # Calculate inverse markers for source
        assert A.shape[1] == len(vertices) - len(markers)
        assert A.shape[0] == b.shape[0]

        LU = sparse_linalg.splu((A.T @ A).tocsc())
        x = LU.solve(A.T @ b)

        # Reconstruct vertices x
        revert_markers(A, x, target, markers, out=vertices)

        result = meshlib.Mesh(vertices=vertices[:len(source_org.vertices)],
                              faces=source_org.faces)
        vertices = result.to_fourth_dimension().vertices

        #########################################################
        if plot:
            pbar_next("Plotting")
            MeshPlots.plot_result_merged(
                source_org, target_org, result, markers,
                mesh_kwargs=dict(flatshading=True)
            )
    return np.array(match_triangles(result, target))


def get_correspondence(source_org: meshlib.Mesh, target_org: meshlib.Mesh, markers: np.ndarray,
                       plot=False) -> np.ndarray:
    hashid = hashlib.sha256()
    hashid.update(b"correspondence")
    hashid.update(markers.data)
    hashid.update(source_org.vertices.data)
    hashid.update(source_org.faces.data)
    hashid.update(target_org.vertices.data)
    hashid.update(target_org.faces.data)
    hashid = hashid.hexdigest()

    cache = CorrespondenceCache(suffix="_tri_markers").entry(hashid=hashid)
    matched_triangles = cache.cache(compute_correspondence, source_org, target_org, markers, plot=plot)
    return matched_triangles


if __name__ == "__main__":
    cfg = ConfigFile.load(ConfigFile.Paths.highpoly.horse_camel)
    # Load meshes
    source_org = meshlib.Mesh.load(cfg.source.reference)
    target_org = meshlib.Mesh.load(cfg.target.reference)
    markers = cfg.markers  # List of vertex-tuples (source, target)

    corres = compute_correspondence(source_org, target_org, markers, plot=True)
    MeshPlots.plot_correspondence(source_org, target_org, corres).show(renderer="browser")
