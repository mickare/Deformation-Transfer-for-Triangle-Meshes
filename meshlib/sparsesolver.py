"""
Multithreading sparse solver, that is not needed anymore!
"""
import multiprocessing
import os
from abc import abstractmethod, ABC
from typing import Union, Optional, Dict, Any, Sequence, Callable

import numpy as np

from scipy import sparse
import scipy.sparse.linalg


class ComponentSolver(ABC):
    @abstractmethod
    def __call__(self, A: sparse.spmatrix, b: Union[sparse.spmatrix, np.ndarray],
                 x0: Optional[Union[np.ndarray]] = None) -> np.ndarray:
        ...


class LSMRSolver(ComponentSolver):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, A: sparse.spmatrix, b: Union[sparse.spmatrix, np.ndarray],
                 x0: Optional[Union[np.ndarray]] = None) -> np.ndarray:
        assert A.shape[0] == b.shape[0]

        x0T = None
        if x0 is not None:
            x0T = x0.T.flatten()
            assert A.shape[1] * b.shape[1] == len(x0T)

        # Build diagonal block
        A = A.tocsc()
        BlockA = sparse.block_diag([A for d in range(b.shape[1])], format="csr")
        BlockB = b.T.flatten()
        assert BlockA.shape[0] == BlockB.shape[0]

        lsqr_result = sparse.linalg.lsmr(BlockA, BlockB, x0=x0T, **self.kwargs)
        return lsqr_result[0].reshape((3, -1)).T


def call_solver(solver: str, A: sparse.spmatrix, b: Union[sparse.spmatrix, np.ndarray],
                x0: Optional[np.ndarray] = None,
                kwargs: Optional[Dict[str, Any]] = None):
    kwargs = kwargs or {}

    if solver == "lsqr":
        return sparse.linalg.lsqr(A, b, x0=x0, **kwargs)[0]
    elif solver == "lsmr":
        return sparse.linalg.lsmr(A, b, x0=x0, **kwargs)[0]
    else:
        raise ValueError(f"Invalid solver identifier {solver}")


class BlockComponentSolver(ComponentSolver):
    """
    A block solver for multiple independent components with the same sparse matrix.
    It sets the A matrix into a sparse block diagional.
    """

    def __init__(self, solver="lsqr", **kwargs):
        self.solver = solver
        self.kwargs = kwargs

    def __call__(self, A: sparse.spmatrix, b: Union[sparse.spmatrix, np.ndarray],
                 x0: Optional[Union[np.ndarray]] = None) -> np.ndarray:
        assert A.shape[0] == b.shape[0]

        x0T = None
        if x0 is not None:
            x0T = x0.T.flatten()
            assert A.shape[1] * b.shape[1] == len(x0T)

        # Build diagonal block
        A = A.tocsc()
        BlockA = sparse.block_diag([A for d in range(b.shape[1])], format="csr")
        BlockB = b.T.flatten()
        assert BlockA.shape[0] == BlockB.shape[0]

        # lsqr_result = sparse.linalg.lsqr(BlockA, BlockB, x0=x0T, **self.kwargs)
        # return lsqr_result[0].reshape((3, -1)).T
        return call_solver(self.solver, BlockA, BlockB, x0T, self.kwargs).reshape((3, -1)).T


class ProcessComponentSolver(ComponentSolver):
    """
    A parallel process solver for multiple independent components with the same sparse matrix
    """

    def __init__(self, solver="lsqr", processes: int = -1, **kwargs):
        self.processes = processes
        self.solver = solver
        self.kwargs = kwargs

    def __call__(self, A: sparse.spmatrix, b: Union[sparse.spmatrix, np.ndarray],
                 x0: Optional[Union[np.ndarray]] = None):
        assert A.shape[0] == b.shape[0]

        x0T: Sequence[np.ndarray] = [None] * b.shape[1]
        if x0 is not None:
            assert A.shape[1] == x0.shape[0]
            assert x0.shape[1] == b.shape[1]
            x0T = x0.T

        A = A.tocsc()
        bT = b.T

        processes = self.processes
        if processes <= 0:
            processes = min(os.cpu_count() or 1, len(bT))
        with multiprocessing.Pool(processes) as pool:
            iter_lim = 30000
            return np.transpose(
                pool.starmap(call_solver, [(self.solver, A, bT[c], x0T[c], self.kwargs) for c in range(len(bT))])
            )
