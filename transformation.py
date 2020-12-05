from config import ConfigFile
from utils import TriangleMarkersCache
import meshlib
import hashlib
from correspondence import get_correspondence

cfg = ConfigFile.load(ConfigFile.Paths.highpoly.horse_camel)

#########################################################
# Load meshes

original_source = meshlib.Mesh.from_file_obj(cfg.source.reference)
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
cache = TriangleMarkersCache(suffix="_tri_markers").entry(hashid=hashid)
markers = cache.get()

if markers is None:
    markers = get_correspondence()
else:
    print("Reusing Correspondence")
