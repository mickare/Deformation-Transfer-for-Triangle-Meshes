from config import ConfigFile
from utils import CorrespondenceCache
import meshlib
import hashlib
from correspondence import get_correspondence
import numpy as np

cfg = ConfigFile.load(ConfigFile.Paths.highpoly.horse_camel)

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
markers = cache.get()

if markers is None:
    markers = get_correspondence()
else:
    print("Reusing Correspondence")

source_mesh = original_source.to_fourth_dimension()
target_mesh = original_target.to_fourth_dimension()
source_pose_mesh = original_transformed_source.to_fourth_dimension()


def compute_s():
    v = source_pose_mesh.span
    inv_v = np.linalg.inv(source_mesh.span)
    s = np.matmul(v, inv_v)
    return s


s = compute_s()
