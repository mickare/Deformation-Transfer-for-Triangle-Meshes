from config import ConfigFile
from utils import CorrespondenceCache
import meshlib
import hashlib
from correspondence import get_correspondence, TransformEntry
import numpy as np
from scipy import sparse
import scipy.sparse.linalg
import tqdm

cfg = ConfigFile.load(ConfigFile.Paths.highpoly.horse_camel)
corr_markers = cfg.markers  # List of vertex-tuples (source, target)

#########################################################
# Load meshes

original_source = meshlib.Mesh.from_file_obj(cfg.source.reference)
original_transformed_source = meshlib.Mesh.from_file_obj(cfg.source.poses[0])
original_target = meshlib.Mesh.from_file_obj(cfg.target.reference)

#########################################################
# Load correspondence

hashid = hashlib.sha256()
hashid.update(b"markers")
hashid.update(bytes([len(original_source.vertices.shape)]))
hashid.update(bytes([len(original_source.faces.shape)]))
hashid.update(bytes([len(original_target.vertices.shape)]))
hashid.update(bytes([len(original_target.faces.shape)]))
hashid = hashid.hexdigest()
cache = CorrespondenceCache(suffix="_tri_markers").entry(hashid=hashid)
mapping = cache.get()

if mapping is None:
    mapping = get_correspondence(original_source, original_target, corr_markers)
else:
    print("Reusing Correspondence")

source_mesh = original_source.to_fourth_dimension()
target_mesh = original_target.to_fourth_dimension()
source_pose_mesh = original_transformed_source.to_fourth_dimension()


def compute_s():
    v = source_pose_mesh.span
    inv_v = np.linalg.inv(source_mesh.span)
    vvinv = np.matmul(v, inv_v)
    s = np.transpose(vvinv[mapping[:, 0]], axes=[0, 2, 1])
    return s


s = compute_s()
inv_v_target = np.linalg.inv(target_mesh.span)

shape = (
    s.shape[0] * 3,
    len(target_mesh.vertices)
)

A = sparse.dok_matrix(shape, dtype=np.float)
transforms = [TransformEntry(f, invV) for f, invV in zip(target_mesh.faces[mapping[:, 1]], inv_v_target[mapping[:, 1]])]
for index, Ti in enumerate(tqdm.tqdm(transforms, desc="Building Transformation Matrix")):  # type: int, TransformEntry
    Ti.insert_to(A, row=index * 3)
b = np.concatenate(s)
assert A.shape[0] == b.shape[0]
assert b.shape[1] == 3
assert A.shape[1] == len(target_mesh.vertices)
A = A.tocsc()
A.eliminate_zeros()

LU = sparse.linalg.splu((A.T @ A).tocsc())
x = LU.solve(A.T @ b)
