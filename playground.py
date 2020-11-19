import numpy as np
from scipy.sparse import coo_matrix, dok_matrix
import scipy.linalg as scilinalg
from scipy.sparse.linalg import spilu, svds, lsqr
from scipy.sparse.linalg.dsolve._superlu import SuperLU

import meshlib
from discrete_mesh import TriangleSpanMesh
from render import get_markers, BrowserVisualizer

cat = "models/lowpoly_cat/cat_reference.obj"
dog = "models/lowpoly_dog/dog_reference.obj"

source = meshlib.Mesh.from_file_obj(cat)
target = meshlib.Mesh.from_file_obj(dog)
markers = get_markers()  # cat, dog
# markers = np.transpose((markers[:, 0], markers[:, 0]))

source_span = TriangleSpanMesh.from_mesh(source)
target_span = TriangleSpanMesh.from_mesh(target)

# Identity
AEi = dok_matrix(
    (len(source_span.faces) * 9, len(source_span.vertices) * 3),
    dtype=np.float
)

b = np.tile(np.identity(3, dtype=np.float).flatten(), len(source_span.faces))
assert AEi.shape[0] == len(b)

invVs = np.linalg.inv(source_span.span)
assert len(source_span.faces) == len(invVs)
# row = 0
for index, (f, invV) in enumerate(zip(source_span.faces, invVs)):  # type: int, (np.ndarray, np.ndarray)
    # Index
    i0 = f[0]
    i1 = f[1]
    i2 = f[2]
    i3 = f[3]

    """
    V = [v2-v1, v3-v1, v4-v1]^-1

    w = xV

    w_ij = v1 x2 - v1 x1 + v2 x3 - v2 x1 + v3 x4 - v3 x1
    w_ij = -(v1+v2+v3) x1 + v1 x2 + v2 x3 + v3 x4
    """
    kleinA = np.zeros(shape=(9, 12))

    # Build T = V~ V^-1
    for i in range(3):  # Row of T
        for j in range(3):  # Column of T
            r = 3 * j + i
            kleinA[r, i] = - (invV[0, j] + invV[1, j] + invV[2, j])
            kleinA[r, i + 3] = invV[0, j]
            kleinA[r, i + 6] = invV[1, j]
            kleinA[r, i + 9] = invV[2, j]

    row = index * 9
    AEi[row:row + 9, i0:i0 + 3] = kleinA[:, 0:3]
    AEi[row:row + 9, i1:i1 + 3] = kleinA[:, 3:6]
    AEi[row:row + 9, i2:i2 + 3] = kleinA[:, 6:9]
    AEi[row:row + 9, i3:i3 + 3] = kleinA[:, 9:12]

for mark_src_i, mark_dest_i in markers:
    i = mark_src_i * 3
    valueB = AEi[:, i:i + 3] @ target_span.vertices[mark_dest_i]
    b -= valueB
    # AEi[:, i:i + 3] = 0

A = AEi.tocsc()
At = A.transpose()
AtA = At @ A
AtB = At @ b

# U, S, Vt = svds(AtA, k=len(source.vertices), which='LM')
# U, S, Vt = svds(AtA, k=len(source.vertices), which='SM')
# result = psInv @ AtB

# U, S, Vt = svds(AtA)
# psInv = Vt.T @ np.linalg.inv(np.diag(S)) @ U.T
# result = psInv @ AtB

lsqr_result = lsqr(A, b)
result = lsqr_result[0]

vertices = result.reshape((-1, 3))

assert vertices.shape[0] == source_span.vertices.shape[0]
final_mesh = meshlib.Mesh(
    vertices=vertices[:, :3],
    faces=source_span.faces[:, :3]
)
# for mark_src_i, mark_dest_i in markers:
#     final_mesh.vertices[mark_src_i] = target.vertices[mark_dest_i]

vis = BrowserVisualizer()
vis.addMesh(final_mesh)
vis.show()

# psInv = spilu(vt * sinv * u.transpose())
# source_vert_flat = source_vert.flatten()
# target_vert_flat = target_vert.flatten()


# adjecent = np.where(source.faces == f)[0]
# for adj in adjecent:
#     if adj != index:
#         set_vertex_index(row, index, invV.transpose())
#         set_vertex_index(row, adj, -invVs[adj].transpose())
#         row += 1
#
# np.transpose(invV)
