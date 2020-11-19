import math
from typing import Union, Sequence, Tuple

import numpy as np

Vec3f = Union[np.ndarray, Sequence[float], Tuple[float, float, float]]


class Vector3D:

    @classmethod
    def new_rotation(cls, axis: Vec3f, angle: float):
        """
            Returns a rotation matrix for a quaternion

            From: https://stackoverflow.com/questions/6802577/rotation-of-3d-vector
            """
        axis = np.asarray(axis)
        axis = axis / math.sqrt(np.dot(axis, axis))
        a = math.cos(angle / 2.0)
        b, c, d = -axis * math.sin(angle / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac), 0],
                         [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab), 0],
                         [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc, 0],
                         [0, 0, 0, 1]])

    @classmethod
    def new_offset(cls, offset: Vec3f):
        offset = np.asarray(offset)
        assert offset.shape == (3,)
        result = np.zeros((4, 4))
        result[0:3, 3] = offset
        return result

    @classmethod
    def rotate(cls, vec: np.ndarray, axis: Vec3f, angle: float):
        return cls.apply(vec, cls.new_rotation(axis, angle))

    @classmethod
    def to_quaternion(cls, vec: np.ndarray) -> np.ndarray:
        assert vec.ndim == 2 and vec.shape[1] == 3
        return np.pad(vec, ((0, 0), (0, 1)), constant_values=1)

    @classmethod
    def apply(cls, vec: np.ndarray, transf: np.ndarray):
        qvec = cls.to_quaternion(vec)
        return np.dot(transf, qvec.T).T[:, :3]
