"""
Caches intermediate results on disk.

It improves repeated execution significantly, especially when debugging things.
"""
import hashlib
import os
from dataclasses import dataclass
from typing import Tuple, Callable, Optional, Union, Sequence

import numpy as np
from scipy import sparse

from .mesh import Mesh


class SparseMatrixCache:
    hashid: str = property(fget=lambda self: self._hashid)
    shape: Tuple[int, int] = property(fget=lambda self: self._shape)

    def __init__(self, suffix='', prefix='', path='.cache'):
        self.suffix = suffix
        self.prefix = prefix
        self.path = path

    @dataclass
    class Entry:
        parent: "SparseMatrixCache"
        hashid: str
        shape: Tuple[int, ...]

        @property
        def file(self):
            return os.path.join(self.parent.path, f"{self.parent.prefix}{self.hashid}{self.parent.suffix}.npz")

        def get(self) -> Optional[sparse.spmatrix]:
            file = self.file
            # Try to load file
            if os.path.isfile(file):
                data = sparse.load_npz(file)
                if data.shape == self.shape:
                    return data
            return None

        def store(self, data: sparse.spmatrix):
            file = self.file
            os.makedirs(os.path.dirname(file), exist_ok=True)
            sparse.save_npz(file, data)

        def cache(self, func: Callable[[], sparse.spmatrix]):
            data = self.get()
            if data is None:
                data = func()
                self.store(data)
            return data

    def entry(self, hashid: str, shape: Tuple[int, ...]):
        assert hashid
        assert shape
        return SparseMatrixCache.Entry(self, hashid, shape)


class DeformedMeshCache:
    def __init__(self, suffix='', prefix='', path='.cache'):
        self.suffix = suffix
        self.prefix = prefix
        self.path = path

    @dataclass
    class Entry:
        parent: "DeformedMeshCache"
        hashid: str
        original: Mesh

        @property
        def file(self):
            return os.path.join(self.parent.path, f"{self.parent.prefix}{self.hashid}{self.parent.suffix}.npz")

        def get(self) -> Optional[Mesh]:
            file = self.file
            # Try to load file
            if os.path.isfile(file):
                with np.load(file) as data:
                    vertices = data["vertices"]
                    faces = data["faces"]
                    if vertices.shape == self.original.vertices.shape and (faces == self.original.faces).all():
                        return Mesh(vertices, faces)
            return None

        def store(self, mesh: Mesh):
            file = self.file
            os.makedirs(os.path.dirname(file), exist_ok=True)
            m = mesh.to_third_dimension()
            np.savez_compressed(file, vertices=m.vertices, faces=m.faces)

        def cache(self, func: Callable[[], Mesh]) -> Mesh:
            mesh = self.get()
            if mesh is None:
                mesh = func()
                self.store(mesh)
            return mesh

    def entry(self, original: Mesh, salts: Sequence[Union[bytes, bytearray, memoryview]] = ()):
        assert original
        h = hashlib.sha256()
        h.update(original.vertices.data)
        h.update(original.faces.data)
        for s in salts:
            h.update(s)
        return DeformedMeshCache.Entry(self, h.hexdigest(), original)


class CorrespondenceCache:
    hashid: str = property(fget=lambda self: self._hashid)

    def __init__(self, suffix='', prefix='', path='.cache'):
        self.suffix = suffix
        self.prefix = prefix
        self.path = path

    @dataclass
    class Entry:
        parent: "CorrespondenceCache"
        hashid: str

        @property
        def file(self):
            return os.path.join(self.parent.path, f"{self.parent.prefix}{self.hashid}{self.parent.suffix}.npz")

        def get(self) -> Optional[np.ndarray]:
            file = self.file
            # Try to load file
            if os.path.isfile(file):
                data = np.load(file)
                return data["correspondence"]
            return None

        def store(self, data: np.ndarray):
            file = self.file
            os.makedirs(os.path.dirname(file), exist_ok=True)
            np.savez_compressed(file, correspondence=data)

        def cache(self, func: Callable[..., np.ndarray], *args, **kwargs):
            data = self.get()
            if data is None:
                data = func(*args, **kwargs)
                self.store(data)
            return data

    def entry(self, hashid: str):
        assert hashid
        return CorrespondenceCache.Entry(self, hashid)
