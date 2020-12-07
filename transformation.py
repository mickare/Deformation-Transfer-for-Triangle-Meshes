import numpy as np
import scipy.sparse.linalg
import tqdm
from scipy import sparse

import meshlib
from config import ConfigFile
from correspondence import get_correspondence, TransformEntry, compute_adjacent_by_edges


class Transformation:
    def __init__(self, source: meshlib.Mesh, target: meshlib.Mesh, mapping: np.ndarray, smoothness=0.00001):
        self.source = source.to_third_dimension(copy=False)
        self.target = target.to_third_dimension(copy=False)
        self.mapping = mapping
        self.Wm = 1.0
        self.Ws = smoothness

        self._Am = self._compute_mapping_matrix(self.target, mapping)
        self._As, self._Bs = self._compute_missing_smoothness(self.target, mapping)

    @classmethod
    def _compute_mapping_matrix(cls, target: meshlib.Mesh, mapping: np.ndarray):
        target = target.to_fourth_dimension(copy=False)
        inv_target_span = np.linalg.inv(target.span)

        # Matrix
        Am = sparse.dok_matrix((
            len(mapping) * 3,
            len(target.vertices)
        ), dtype=np.float)

        transforms = [TransformEntry(f, invV) for f, invV in
                      zip(target.faces[mapping[:, 1]], inv_target_span[mapping[:, 1]])]
        for index, Ti in enumerate(
                tqdm.tqdm(transforms, desc="Building Mapping")):  # type: int, TransformEntry
            Ti.insert_to(Am, row=index * 3)

        return Am.tocsc()

    @classmethod
    def _compute_missing_smoothness(cls, target: meshlib.Mesh, mapping: np.ndarray):
        adjacent = compute_adjacent_by_edges(target)
        target = target.to_fourth_dimension(copy=False)
        inv_target_span = np.linalg.inv(target.span)
        missing = np.setdiff1d(np.arange(len(target.faces)), np.unique(mapping[:, 1]))
        count_adjacent = sum(len(adjacent[m]) for m in missing)
        shape = (
            count_adjacent * 3,
            len(target.vertices)
        )
        transforms_all = [(n, TransformEntry(f, invV)) for n, (f, invV) in
                          enumerate(zip(target.faces, inv_target_span))]
        transforms_mis = [transforms_all[m] for m in missing]

        lhs = sparse.dok_matrix(shape, dtype=np.float)
        rhs = sparse.dok_matrix(shape, dtype=np.float)
        row = 0
        for index, Ti in tqdm.tqdm(transforms_mis, desc="Building Smoothness"):
            for adj in adjacent[index]:
                Ti.insert_to(lhs, row)
                transforms_all[adj][1].insert_to(rhs, row)
                row += 3
        As = (lhs - rhs)
        Bs = np.zeros((As.shape[0], 3))
        return As, Bs

    def __call__(self, pose: meshlib.Mesh, smoothness=0.00001) -> meshlib.Mesh:
        assert smoothness > 0
        # Transformation of source
        ## Si * V = V~  ==>>  Si = V~ * V^-1
        s = np.linalg.inv(self.source.span) @ pose.span
        ## Stack Si
        Bm = np.concatenate(s[self.mapping[:, 0]])

        Astack = [self._Am * self.Wm, self._As * self.Ws]
        Bstack = [Bm * self.Wm, self._Bs * self.Ws]

        A: sparse.spmatrix = sparse.vstack(Astack, format="csc")
        A.eliminate_zeros()
        b = np.concatenate(Bstack)

        assert A.shape[0] == b.shape[0]
        assert b.shape[1] == 3
        # assert A.shape[1] == len(target_mesh.vertices)
        A = A.tocsc()
        A.eliminate_zeros()

        LU = sparse.linalg.splu((A.T @ A).tocsc())
        x = LU.solve(A.T @ b)

        result = meshlib.Mesh(vertices=x[:len(self.target.vertices)], faces=self.target.faces)
        return result



if __name__ == "__main__":
    import plot_result
    import render

    # cfg = ConfigFile.load(ConfigFile.Paths.highpoly.horse_camel)
    cfg = ConfigFile.load(ConfigFile.Paths.lowpoly.catdog)
    # cfg = ConfigFile.load(ConfigFile.Paths.highpoly.cat_lion)
    corr_markers = cfg.markers  # List of vertex-tuples (source, target)

    identity = False
    if identity:
        corr_markers = np.ascontiguousarray(np.array((corr_markers[:, 0], corr_markers[:, 0]), dtype=np.int).T)

    #########################################################
    # Load meshes

    original_source = meshlib.Mesh.from_file_obj(cfg.source.reference)
    original_pose = meshlib.Mesh.from_file_obj(cfg.source.poses[0])
    original_target = meshlib.Mesh.from_file_obj(cfg.target.reference)
    if identity:
        original_target = meshlib.Mesh.from_file_obj(cfg.source.reference)

    #########################################################
    # Load correspondence from cache if possible
    mapping = get_correspondence(original_source, original_target, corr_markers, plot=True)

    transf = Transformation(original_source, original_target, mapping)
    result = transf(original_pose)

    render.MeshPlots.plot_correspondence(original_source, original_target, mapping)
    plot_result.plot(original_pose, result).show(renderer="browser")
