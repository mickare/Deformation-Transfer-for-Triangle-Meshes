import os
from abc import abstractmethod, ABC
from dataclasses import dataclass
from functools import reduce
from typing import Tuple, Callable, Optional, Any

from scipy import sparse


def tween(seq, sep):
    """From: https://stackoverflow.com/questions/5920643/add-an-item-between-each-item-already-in-the-list"""
    return reduce(lambda r, v: r + [sep, v], seq[1:], seq[:1])


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
