from typing import List

import numpy as np
import scipy
from scipy.sparse import dok_matrix, lil_matrix
from scipy.sparse.linalg import lsqr

import meshlib
from discrete_mesh import TriangleSpanMesh
from render import get_markers, BrowserVisualizer
from utils import tween

cat = "models/lowpoly_cat/cat_reference.obj"
dog = "models/lowpoly_dog/dog_reference.obj"

original_source = meshlib.Mesh.from_file_obj(cat)
original_target = meshlib.Mesh.from_file_obj(dog)
markers = get_markers()  # cat, dog
# markers = np.transpose((markers[:, 0], markers[:, 0]))

target_mesh = TriangleSpanMesh.from_mesh(original_target)
subject = TriangleSpanMesh.from_mesh(original_source)

# Weights of cost functions
Ws = 1.0
Wi = 0.001

# Precalculate the adjacent triangles in source
print("Prepare adjacent list")
adjacent: List[List[int]] = [[j for j in np.where(subject.faces == f)[0] if j != i]
                             for i, f in enumerate(subject.faces)]


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

    def insert_to(self, target, row: int, factor=1.0):
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
# Start of loop

# Create inverse of triangle spans
print("Inverse Triangle Spans")
invVs = np.linalg.inv(subject.span)
assert len(subject.faces) == len(invVs)

# Preparing the transformation matrices
print("Preparing Transforms")
transforms = [TransformEntry(f, invV) for f, invV in zip(subject.faces, invVs)]

#########################################################
# Identity Cost - of transformations
print("Building Identity Cost")
Bi = np.tile(np.identity(3, dtype=np.float).flatten(), len(subject.faces))
AEi = lil_matrix(
    (
        # Count of all minimization terms
        len(subject.faces) * 9,
        # Length of flat result x
        len(subject.vertices) * 3
    ),
    dtype=np.float
)
assert AEi.shape[0] == len(Bi)
for index, Ti in enumerate(transforms):  # type: int, (np.ndarray, np.ndarray)
    Ti.insert_to(AEi, row=index * 9)

#########################################################
# Smoothness Cost - of differences to adjacent transformations
print("Building Smoothness Cost")
count_adjacent = sum(len(a) for a in adjacent)
Bs = np.zeros(count_adjacent * 9)
AEs = lil_matrix(
    (
        # Count of all minimization terms
        count_adjacent * 9,
        # Length of flat result x
        len(subject.vertices) * 3
    ),
    dtype=np.float
)
assert AEs.shape[0] == len(Bs)
row = 0
for index, Ti in enumerate(transforms):  # type: int, (np.ndarray, np.ndarray)
    for adj in adjacent[index]:
        Ti.insert_to(AEs, row)
        transforms[adj].insert_to(AEs, row, -1.0)
        row += 1
assert row == count_adjacent

#########################################################
print("Combining Costs")
A = scipy.sparse.vstack((AEi, AEs), format="lil")
b = np.concatenate((Wi * Bi, Ws * Bs))

print("Enforcing Markers")
for mark_src_i, mark_dest_i in markers:
    i = mark_src_i * 3
    valueB = Atmp[:, i:i + 3] @ target_mesh.vertices[mark_dest_i]
    b -= valueB
    Atmp[:, i:i + 3] = 0

#########################################################
print("Solving")
# U, S, Vt = svds(A)
# psInv = Vt.T @ np.linalg.inv(np.diag(S)) @ U.T
# result = psInv @ b
lsqr_result = lsqr(A, b)
result = lsqr_result[0]

# Apply new vertices
print("Appling vertices")
vertices = result.reshape((-1, 3))
old_subject = subject
subject = TriangleSpanMesh.from_parts(vertices=vertices, faces=original_source.faces)
# Enforce target vertices
for mark_src_i, mark_dest_i in markers:
    subject.vertices[mark_src_i] = target_mesh.vertices[mark_dest_i]

#########################################################
print("Rendering")

vis = BrowserVisualizer()
vis.addMesh(subject.to_mesh())
vis.addScatter(
    tween(original_target.vertices.tolist(), (np.nan, np.nan, np.nan)),
    marker=dict(
        color='red',
        size=3
    )
)
vis.show()
