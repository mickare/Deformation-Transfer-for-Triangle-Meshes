from config import ConfigFile
from plot_result import plot_result
from utils import CorrespondenceCache
import meshlib
import hashlib
from correspondence import get_correspondence, TransformEntry, compute_adjacent_by_edges
import numpy as np
from scipy import sparse
import scipy.sparse.linalg
import tqdm

cfg = ConfigFile.load(ConfigFile.Paths.highpoly.horse_camel)
# cfg = ConfigFile.load(ConfigFile.Paths.lowpoly.catdog)
corr_markers = cfg.markers  # List of vertex-tuples (source, target)

# corr_markers = np.ascontiguousarray(np.array((corr_markers[:, 0], corr_markers[:, 0]), dtype=np.int).T)

#########################################################
# Load meshes

original_source = meshlib.Mesh.from_file_obj(cfg.source.reference)
original_transformed_source = meshlib.Mesh.from_file_obj(cfg.source.poses[0])
original_target = meshlib.Mesh.from_file_obj(cfg.target.reference)
# original_target = meshlib.Mesh.from_file_obj(cfg.source.reference)

source_mesh = original_source.to_fourth_dimension()
target_mesh = original_target.to_fourth_dimension()
pose_mesh = original_transformed_source.to_fourth_dimension()

#########################################################
# Load correspondence from cache if possible
mapping = get_correspondence(original_source, original_target, corr_markers, plot=True)


#########################################################
# Prepare transformation matrices

def compute_s():
    v = source_mesh.span
    inv_v = np.linalg.inv(pose_mesh.span)
    vvinv = v.transpose((0,2,1)) @ inv_v.transpose((0,2,1))
    return vvinv[mapping[:, 0]]


inv_target_span = np.linalg.inv(target_mesh.span)


# Mapping matrix
def build_mapping():
    s = compute_s()
    Am = sparse.dok_matrix((
        s.shape[0] * 3,
        len(target_mesh.vertices)
    ), dtype=np.float)

    transforms = [TransformEntry(f, invV) for f, invV in
                  zip(target_mesh.faces[mapping[:, 1]], inv_target_span[mapping[:, 1]])]

    for index, Ti in enumerate(
            tqdm.tqdm(transforms, desc="Building Mapping")):  # type: int, TransformEntry
        Ti.insert_to(Am, row=index * 3)

    Bm = np.concatenate(s)

    return Am, Bm


# Smoothness over missing mappings
def build_missing():
    missing = np.setdiff1d(np.arange(len(target_mesh.faces)), np.unique(mapping[:, 1]))
    adjacent = compute_adjacent_by_edges(original_target)
    count_adjacent = sum(len(adjacent[m]) for m in missing)
    shape = (
        count_adjacent * 3,
        len(target_mesh.vertices)
    )
    transforms_all = [(n, TransformEntry(f, invV)) for n, (f, invV) in
                      enumerate(zip(target_mesh.faces, inv_target_span))]
    transforms_mis = [transforms_all[m] for m in missing]

    lhs = sparse.dok_matrix(shape, dtype=np.float)
    rhs = sparse.dok_matrix(shape, dtype=np.float)
    row = 0
    for index, Ti in tqdm.tqdm(transforms_mis, desc="Building Smoothness"):
        for adj in adjacent[index]:
            Ti.insert_to(lhs, row)
            transforms_all[adj][1].insert_to(rhs, row)
            row += 3
    As = (lhs - rhs)
    Bs = np.zeros((As.shape[0], 3))
    return As, Bs


Am, Bm = build_mapping()
As, Bs = build_missing()

Wm = 1.0
Ws = 1.0
Astack = [Am * Wm, As * Ws]
Bstack = [Bm * Wm, Bs * Ws]

A: sparse.spmatrix = sparse.vstack(Astack, format="csc")
A.eliminate_zeros()
b = np.concatenate(Bstack)

assert A.shape[0] == b.shape[0]
assert b.shape[1] == 3
assert A.shape[1] == len(target_mesh.vertices)
A = A.tocsc()
A.eliminate_zeros()

LU = sparse.linalg.splu((A.T @ A).tocsc())
x = LU.solve(A.T @ b)

result = meshlib.Mesh(vertices=x[:len(original_target.vertices)], faces=original_target.faces)
plot_result(pose_mesh, result).show(renderer="browser")
