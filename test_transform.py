from config import ConfigFile
import plot_result
from utils import CorrespondenceCache
import meshlib
import hashlib
from correspondence import get_correspondence, TransformEntry, compute_adjacent_by_edges
import numpy as np
from scipy import sparse
import scipy.sparse.linalg
import tqdm

# cfg = ConfigFile.load(ConfigFile.Paths.highpoly.horse_camel)
# cfg = ConfigFile.load(ConfigFile.Paths.lowpoly.catdog)
cfg = ConfigFile.load(ConfigFile.Paths.highpoly.cat_lion)
corr_markers = cfg.markers  # List of vertex-tuples (source, target)

identity = False
if identity:
    corr_markers = np.ascontiguousarray(np.array((corr_markers[:, 0], corr_markers[:, 0]), dtype=np.int).T)

#########################################################
# Load meshes

original_source = meshlib.Mesh.load(cfg.source.reference)
original_transformed_source = meshlib.Mesh.load(cfg.source.poses[0])
original_target = meshlib.Mesh.load(cfg.target.reference)
if identity:
    original_target = meshlib.Mesh.load(cfg.source.reference)

source_mesh = original_source.to_fourth_dimension()
target_mesh = original_target.to_fourth_dimension()
pose_mesh = original_transformed_source.to_fourth_dimension()

#########################################################
# Load correspondence from cache if possible
mapping = get_correspondence(original_source, original_target, corr_markers, plot=True)


#########################################################
# Prepare transformation matrices

def compute_s():
    # Si * V = V~  ==>>  Si = V~ * V^-1
    return np.linalg.inv(source_mesh.span) @ pose_mesh.span


inv_target_span = np.linalg.inv(target_mesh.span)


# Mapping matrix
def build_mapping():
    Bm = np.concatenate(compute_s()[mapping[:, 0]])
    Am = sparse.dok_matrix((
        Bm.shape[0],
        len(target_mesh.vertices)
    ), dtype=np.float)

    transforms = [TransformEntry(f, invV) for f, invV in
                  zip(target_mesh.faces[mapping[:, 1]], inv_target_span[mapping[:, 1]])]

    for index, Ti in enumerate(
            tqdm.tqdm(transforms, desc="Building Mapping")):  # type: int, TransformEntry
        Ti.insert_to(Am, row=index * 3)

    return Am.tocsc(), Bm


faces_unique = np.unique(target_mesh.faces[mapping[:, 1]].flatten())
missing_verts = np.setdiff1d(np.arange(len(target_mesh.vertices)), faces_unique)
verts_index = np.sort(faces_unique)

Am, Bm = build_mapping()

print("Reduce")
A = Am[:, verts_index].tocsc()
b = Bm

print("Solve")

assert A.shape[0] == b.shape[0]
assert b.shape[1] == 3
# assert A.shape[1] == len(target_mesh.vertices)
A = A.tocsc()
A.eliminate_zeros()

LU = sparse.linalg.splu((A.T @ A).tocsc())
x = LU.solve(A.T @ b)

print("Return")
vertices = np.zeros((len(target_mesh.vertices), 3))
vertices[missing_verts] = np.inf
vertices[verts_index] = x

print("plot")
result = meshlib.Mesh(vertices=vertices[:len(original_target.vertices)], faces=original_target.faces[mapping[:, 1]])
plot_result.plot(pose_mesh, result).show(renderer="browser")
