import hashlib
from collections import defaultdict
from typing import Tuple, Dict, Set

import numpy as np
import scipy.sparse.linalg
import tqdm
from scipy import sparse
from scipy.spatial import KDTree

import meshlib
from config import ConfigFile
from render import MeshPlots
from utils import SparseMatrixCache

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


#########################################################
# Closest point search

def get_aec(columns, rows):
    return sparse.identity(columns, dtype=np.float, format="csc")[:rows]


def get_bec(closest_points: np.array, verts: np.array):
    return verts[closest_points]


def get_closest_points(kd_tree: KDTree, verts: np.array, vert_normals: np.array, target_normals: np.array):
    closest_points = []
    max_angle = np.radians(90)
    for v in range(len(verts)):
        valid = False
        i = 200
        while not valid:
            neighbours = kd_tree.query(verts[v], i)
            for n in neighbours[1]:
                angle = np.arccos(np.dot(vert_normals[v], target_normals[n]))
                if angle < max_angle:
                    valid = True
                    closest_points.append(n)
                    break
            i += 1000
            if i > len(verts):
                closest_points.append(
                    kd_tree.query(verts[v], 1)[1])  # ignore 90 degree restriction if no valid point exists
                break
    return closest_points


def get_vertex_normals(verts: np.array, faces: np.array):
    candidates = [set() for i in range(len(verts))]
    for n, f in enumerate(faces):
        f0, f1, f2, f3 = f
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
    return vertex_normals


def get_triangle_normals(verts: np.array, faces: np.array):
    vns = np.cross(verts[faces[:, 1]] - verts[faces[:, 0]], verts[faces[:, 2]] - verts[faces[:, 0]])
    return (vns.T / np.linalg.norm(vns, axis=1)).T


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


AEi, Bi = construct_identity_cost(subject, transforms)


#########################################################
# Smoothness Cost - of differences to adjacent transformations


def construct_smoothness_cost(subject, transforms, adjacent) -> Tuple[sparse.spmatrix, np.ndarray]:
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


AEs, Bs = construct_smoothness_cost(subject, transforms, adjacent)

#########################################################
print("Building KDTree for closest points")
# KDTree for closest points in E_c
kd_tree_target = KDTree(target_mesh.vertices)
target_normals = get_vertex_normals(target_mesh.vertices, target_mesh.faces)
vertices: np.ndarray = np.copy(subject.vertices)

#########################################################
# Start of loop

iterations = len(Wc)
total_steps = 6  # Steps per iteration
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
        closest_points = get_closest_points(kd_tree_target, vertices[:len(original_source.vertices)],
                                            get_vertex_normals(vertices, subject.faces), target_normals)
        Bc = get_bec(closest_points, target_mesh.vertices)
        assert AEc.shape[0] == Bc.shape[0]
        Astack.append(AEc * Wc[iteration])
        Bstack.append(Bc * Wc[iteration])

    #########################################################
    pbar_next("Combining Costs")

    A: sparse.spmatrix = sparse.vstack(Astack, format="csc")
    A.eliminate_zeros()
    b = np.concatenate(Bstack)
    # b = np.vstack(Bstack)

    #########################################################
    pbar_next("Enforcing Markers")
    for mark_src_i, mark_dest_i in markers:
        valueB = A[:, mark_src_i] * target_mesh.vertices[mark_dest_i].reshape((-1, 3))
        b -= valueB

    A = A.tolil()
    for mark_src_i, mark_dest_i in markers:
        A[:, mark_src_i] = 0

    #########################################################
    pbar_next("Solving")
    A = A.tocsc()

    # Calculate inverse markers for source
    invmarker = np.setdiff1d(np.arange(A.T.shape[0]), markers[:, 0])

    # Compute LU for AtA without markers
    Z = A[:, invmarker].tocsc()
    ZtZ = (Z.T @ Z).tocsc()
    ZtZ.eliminate_zeros()
    LU = sparse.linalg.splu(ZtZ)

    # Solve for x without markers
    zb = LU.solve(Z.T @ b)

    # Reconstruct vertices x
    vertices[invmarker] = zb
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
