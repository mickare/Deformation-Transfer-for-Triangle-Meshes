import hashlib
from collections import defaultdict
from typing import Tuple, Dict, Set

import numpy as np
import scipy.sparse.linalg
import tqdm
from scipy import sparse
from scipy.spatial import cKDTree

import meshlib
from config import ConfigFile
from render import MeshPlots
from utils import SparseMatrixCache
from utils import CorrespondenceCache


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
    return sparse.identity(columns, dtype=np.float, format="csc")[:rows]


def get_bec(closest_points: np.array, verts: np.array):
    return verts[closest_points]


def fallback_closest_points(kd_tree: cKDTree, vert: np.ndarray, normal: np.ndarray, target_normals: np.ndarray,
                            max_angle: float = np.radians(90)) -> int:
    for i in range(int(np.ceil(len(target_normals) / 1000))):
        start = i * 1000 + 1
        ks = np.arange(start, min(start + 1000, len(target_normals)))
        dist, ind = kd_tree.query(vert, ks)
        angles = np.arccos(np.dot(target_normals[ind], normal))
        angles_cond = angles < max_angle
        if angles_cond.any():
            return ind[angles_cond][0]
    raise RuntimeError("Could not find any point on the target mesh!")


def get_closest_points(kd_tree: cKDTree, verts: np.array, vert_normals: np.array, target_normals: np.array,
                       max_angle: float = np.radians(90)) -> np.ndarray:
    assert len(verts) == len(vert_normals)
    closest_points = []
    dists, indicies = kd_tree.query(verts, min(len(target_normals), 200))
    for v, (dist, ind) in enumerate(zip(dists, indicies)):
        angles = np.arccos(np.dot(target_normals[ind], vert_normals[v]))
        angles_cond = angles < max_angle
        if angles_cond.any():
            cind = ind[angles_cond][0]
        else:
            # Fallback
            cind = fallback_closest_points(kd_tree, verts[v], vert_normals[v], target_normals, max_angle)
        closest_points.append(cind)
    return np.array(closest_points)


def get_vertex_normals(verts: np.array, faces: np.array) -> np.ndarray:
    candidates = [set() for i in range(len(verts))]
    for n, (f0, f1, f2, f3) in enumerate(faces):
        candidates[f0].add(n)
        candidates[f1].add(n)
        candidates[f2].add(n)

    """
    Normals only point in the correct direction if the vertices are in the right order in faces. This might not hold for 
    meshes created by scanners.
    """
    triangle_normals = get_triangle_normals(verts, faces)
    triangle_normals_per_vertex = [[triangle_normals[i] for i in indices] for indices in candidates]
    vertex_normals = [np.mean(normals, 0) if normals else np.zeros(3, float) for normals in triangle_normals_per_vertex]
    return np.array(vertex_normals)


def get_triangle_normals(verts: np.array, faces: np.array):
    vns = np.cross(verts[faces[:, 1]] - verts[faces[:, 0]], verts[faces[:, 2]] - verts[faces[:, 0]])
    return (vns.T / np.linalg.norm(vns, axis=1)).T


def max_triangle_length(mesh: meshlib.Mesh):
    a, b, c = mesh.span_components()
    return max(np.max(np.linalg.norm(a, axis=1)), np.max(np.linalg.norm(b, axis=1)))


def match_triangles(source: meshlib.Mesh, target: meshlib.Mesh):
    source_centroids = source.get_centroids()
    target_centroids = target.get_centroids()
    source_normals = source.normals()
    target_normals = target.normals()
    radius = max(max_triangle_length(source), max_triangle_length(target)) * 2
    triangles = get_closest_triangles(source_normals, target_normals, source_centroids, target_centroids, radius)
    tmp_triangles = get_closest_triangles(target_normals, source_normals, target_centroids, source_centroids,
                                          radius)
    triangles.update((t[1], t[0]) for t in tmp_triangles)
    return triangles


def get_closest_triangles(
        source_normals: np.ndarray,
        target_normals: np.ndarray,
        source_centroids: np.ndarray,
        target_centroids: np.ndarray,
        max_angle: float = np.radians(90),
        k: int = 200,
        radius: float = np.inf
) -> Set[Tuple[int, int]]:
    assert len(source_normals) == len(source_centroids)
    assert len(target_normals) == len(target_centroids)
    triangles = set()
    kd_tree = cKDTree(target_centroids)

    dists, indicies = kd_tree.query(source_centroids, min(len(target_centroids), k), distance_upper_bound=radius)
    for t, (dist, ind) in enumerate(zip(dists, indicies)):
        angles = np.arccos(np.dot(target_normals[ind], source_normals[t]))
        angles_cond = angles < max_angle
        if angles_cond.any():
            cind = ind[angles_cond][0]
            triangles.add((t, cind))

    return triangles


#########################################################
# Matrix builder for T Transformation entries

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
        w_ij = -(v1j+v2j+v3j) x1i + (v1j) x2i + (v2j) x3i + (v3j) x4i
        """
        self.invV = invV

    def insert_to(self, target: sparse.spmatrix, row: int):
        # Index
        i0, i1, i2, i3 = self.face
        # Insert by adding
        tmp = self.invV.T
        target[row:row + 3, i0] = -np.sum(tmp, axis=0).reshape((3, 1))
        target[row:row + 3, i1] = tmp[0].reshape((3, 1))
        target[row:row + 3, i2] = tmp[1].reshape((3, 1))
        target[row:row + 3, i3] = tmp[2].reshape((3, 1))


def enforce_markers(A: sparse.spmatrix, b: np.ndarray, target: meshlib.Mesh, markers: np.ndarray) \
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
    invmarker = np.setdiff1d(np.arange(A.shape[1]), markers[:, 0])
    zb = b - A[:, markers.T[0]] * target.vertices[markers.T[1]]
    return A[:, invmarker].tocsc(), zb


#########################################################
# Identity Cost - of transformations


def construct_identity_cost(subject, transforms) -> Tuple[sparse.spmatrix, np.ndarray]:
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
        AEi = sparse.dok_matrix(shape, dtype=np.float)
        for index, Ti in enumerate(tqdm.tqdm(transforms, desc="Building Identity Cost")):  # type: int, TransformEntry
            Ti.insert_to(AEi, row=index * 3)
        AEi = AEi.tocsr()
        AEi.eliminate_zeros()
        cache.store(AEi)
    else:
        print("Reusing Identity Cost")

    Bi = np.tile(np.identity(3, dtype=np.float), (len(subject.faces), 1))
    assert AEi.shape[0] == Bi.shape[0]
    return AEi.tocsr(), Bi


#########################################################
# Smoothness Cost - of differences to adjacent transformations


def construct_smoothness_cost(subject, transforms, adjacent, AEi) -> Tuple[sparse.spmatrix, np.ndarray]:
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
        lhs = sparse.dok_matrix(shape, dtype=np.float)
        rhs = sparse.dok_matrix(shape, dtype=np.float)
        row = 0
        for index, Ti in enumerate(tqdm.tqdm(transforms, desc="Building Smoothness Cost")):
            for adj in adjacent[index]:
                Ti.insert_to(lhs, row)
                transforms[adj].insert_to(rhs, row)
                row += 3
        AEs = (lhs - rhs)
        assert row == AEs.shape[0]
        AEs = AEs.tocsr()
        AEi.eliminate_zeros()
        cache.store(AEs)
    else:
        print("Reusing Smoothness Cost")

    Bs = np.zeros((count_adjacent * 3, 3))
    assert AEs.shape[0] == Bs.shape[0]
    return AEs, Bs


def get_correspondence():
    #########################################################
    # Configuration

    # Meshes
    # cfg = ConfigFile.load(ConfigFile.Paths.lowpoly.catdog)
    cfg = ConfigFile.load(ConfigFile.Paths.highpoly.horse_camel)

    # Weights of cost functions
    Ws = np.sqrt(1.0)
    Wi = np.sqrt(0.001)
    Wc = np.sqrt([0.0, 1.0, 200.0, 1000.0, 5000.0])

    #########################################################
    # Load meshes
    original_source = meshlib.Mesh.from_file_obj(cfg.source.reference)
    original_target = meshlib.Mesh.from_file_obj(cfg.target.reference)
    markers = cfg.markers  # List of vertex-tuples (source, target)

    target_mesh = original_target.to_fourth_dimension()
    subject = original_source.to_fourth_dimension()
    # Show the source and target
    # MeshPlots.side_by_side([original_source, original_target]).show(renderer="browser")

    #########################################################
    # Precalculate the adjacent triangles in source
    print("Precalculate adjacent list")

    # adjacent = compute_adjacent_by_vertices(original_source)
    adjacent = compute_adjacent_by_edges(original_source)

    #########################################################
    print("Inverse Triangle Spans")
    invVs = np.linalg.inv(subject.span)
    assert len(subject.faces) == len(invVs)

    #########################################################
    # Preparing the transformation matrices
    print("Preparing Transforms")
    transforms = [TransformEntry(f, invV) for f, invV in zip(subject.faces, invVs)]

    AEi, Bi = enforce_markers(*construct_identity_cost(subject, transforms), target_mesh, markers)

    AEs, Bs = enforce_markers(*construct_smoothness_cost(subject, transforms, adjacent, AEi), target_mesh, markers)

    #########################################################
    print("Building KDTree for closest points")
    # KDTree for closest points in E_c
    kd_tree_target = cKDTree(target_mesh.vertices)
    target_normals = get_vertex_normals(target_mesh.vertices, target_mesh.faces)
    vertices: np.ndarray = np.copy(subject.vertices)

    #########################################################
    # Start of loop

    iterations = len(Wc)
    total_steps = 4  # Steps per iteration
    # Progress bar
    pBar = tqdm.tqdm(total=iterations * total_steps)

    for iteration in range(iterations):

        def pbar_next(msg: str):
            pBar.set_description(f"[{iteration + 1}/{iterations}] {msg}")
            pBar.update()

        Astack = [AEi * Wi, AEs * Ws]
        Bstack = [Bi * Wi, Bs * Ws]

        # Astack = [AEs * Ws]
        # Bstack = [Bs * Ws]

        #########################################################
        pbar_next("Closest Point Costs")

        if iteration > 0:
            AEc = get_aec(len(subject.vertices), len(original_source.vertices))
            vertices_clipped = vertices[:len(original_source.vertices)]
            closest_points = get_closest_points(kd_tree_target, vertices_clipped,
                                                get_vertex_normals(vertices_clipped, subject.faces), target_normals)
            Bc = get_bec(closest_points, target_mesh.vertices)
            assert AEc.shape[0] == Bc.shape[0]

            mAEc, mBc = enforce_markers(AEc, Bc, target_mesh, markers)
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
        invmarker = np.setdiff1d(np.arange(len(vertices)), markers[:, 0])
        assert len(vertices) - len(markers) == len(invmarker)
        assert A.shape[1] == len(invmarker)
        assert A.shape[0] == b.shape[0]

        LU = sparse.linalg.splu((A.T @ A).tocsc())
        x = LU.solve(A.T @ b)

        # Reconstruct vertices x
        vertices[invmarker] = x
        vertices[markers[:, 0]] = target_mesh.vertices[markers[:, 1]]

        result = meshlib.Mesh(vertices=vertices[:len(original_source.vertices)],
                              faces=original_source.faces)
        vertices = result.to_fourth_dimension().vertices

        #########################################################
        pbar_next("Plotting")
        MeshPlots.result_merged(
            original_source, original_target, result, markers,
            mesh_kwargs=dict(flatshading=True)
        )

    hashid = hashlib.sha256()
    hashid.update(b"markers")
    hashid.update(bytes([len(original_source.vertices.shape)]))
    hashid.update(bytes([len(original_source.faces.shape)]))
    hashid.update(bytes([len(original_target.vertices.shape)]))
    hashid.update(bytes([len(original_target.faces.shape)]))
    hashid = hashid.hexdigest()

    matched_triangles = match_triangles(result, target_mesh)
    cache = CorrespondenceCache(suffix="_tri_markers").entry(hashid=hashid)
    cache.store(np.array(list(matched_triangles)))
    return matched_triangles
