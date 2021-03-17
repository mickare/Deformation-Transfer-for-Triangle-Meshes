import numpy as np
import tqdm
from scipy import sparse

import meshlib
from config import ConfigFile
from correspondence import get_correspondence, compute_adjacent_by_edges, TransformMatrix


class Transformation:
    def __init__(
            self,
            source: meshlib.Mesh,
            target: meshlib.Mesh,
            mapping: np.ndarray,
            smoothness=1.0
    ):
        self.source = source.to_third_dimension(copy=False)
        self.target = target.to_third_dimension(copy=False)
        self.mapping = mapping
        self.Wm = 1.0
        self.Ws = max(0.00000001, smoothness)

        self._Am = self._compute_mapping_matrix(self.target, mapping)
        self._As, self._Bs = self._compute_missing_smoothness(self.target, mapping)

    @classmethod
    def _compute_mapping_matrix(cls, target: meshlib.Mesh, mapping: np.ndarray):
        target = target.to_fourth_dimension(copy=False)
        inv_target_span = np.linalg.inv(target.span)

        # Matrix
        Am = TransformMatrix.construct(target.faces[mapping[:, 1]], inv_target_span[mapping[:, 1]],
                                       len(target.vertices), desc="Building Mapping")
        return Am.tocsc()

    @classmethod
    def _compute_missing_smoothness(cls, target: meshlib.Mesh, mapping: np.ndarray):
        adjacent = compute_adjacent_by_edges(target)
        target = target.to_fourth_dimension(copy=False)
        inv_target_span = np.linalg.inv(target.span)
        missing = np.setdiff1d(np.arange(len(target.faces)), np.unique(mapping[:, 1]))
        count_adjacent = sum(len(adjacent[m]) for m in missing)
        # shape = (
        #     count_adjacent * 3,
        #     len(target.vertices)
        # )

        if count_adjacent == 0:
            return sparse.csc_matrix((0, len(target.vertices)), dtype=float), np.zeros((0, 3))

        size = len(target.vertices)

        def construct(f, inv, index):
            a = TransformMatrix.expand(f, inv, size).tocsc()
            for adj in adjacent[index]:
                yield a, TransformMatrix.expand(target.faces[adj], inv_target_span[adj], size).tocsc()

        lhs, rhs = zip(*(adjacents for index, m in
                         enumerate(tqdm.tqdm(missing, total=len(missing),
                                             desc="Fixing Missing Mapping with Smoothness"))
                         for adjacents in construct(target.faces[m], inv_target_span[m], index)))

        As = (sparse.vstack(lhs) - sparse.vstack(rhs)).tocsc()
        Bs = np.zeros((As.shape[0], 3))
        return As, Bs

    def __call__(self, pose: meshlib.Mesh) -> meshlib.Mesh:
        # Transformation of source
        ## Si * V = V~  ==>>  Si = V~ * V^-1
        s = (pose.span @ np.linalg.inv(self.source.span)).transpose(0, 2, 1)
        ## Stack Si
        Bm = np.concatenate(s[self.mapping[:, 0]])

        Astack = [self._Am * self.Wm, self._As * self.Ws]
        Bstack = [Bm * self.Wm, self._Bs * self.Ws]

        A: sparse.spmatrix = sparse.vstack(Astack, format="csc")
        A.eliminate_zeros()
        b = np.concatenate(Bstack)

        assert A.shape[0] == b.shape[0]
        assert b.shape[1] == 3
        LU = sparse.linalg.splu((A.T @ A).tocsc())
        x = LU.solve(A.T @ b)

        vertices = x
        result = meshlib.Mesh(vertices=vertices[:len(self.target.vertices)], faces=self.target.faces)
        return result


if __name__ == "__main__":
    import render.plot_result as plt_res
    import render.plot as plt

    # cfg = ConfigFile.load(ConfigFile.Paths.highpoly.horse_camel)
    cfg = ConfigFile.load(ConfigFile.Paths.lowpoly.catdog)
    # cfg = ConfigFile.load(ConfigFile.Paths.highpoly.cat_lion)
    corr_markers = cfg.markers  # List of vertex-tuples (source, target)

    identity = False
    if identity:
        corr_markers = np.ascontiguousarray(np.array((corr_markers[:, 0], corr_markers[:, 0]), dtype=np.int).T)

    #########################################################
    # Load meshes

    original_source = meshlib.Mesh.load(cfg.source.reference)
    original_pose = meshlib.Mesh.load(cfg.source.poses[0])
    original_target = meshlib.Mesh.load(cfg.target.reference)
    if identity:
        original_target = meshlib.Mesh.load(cfg.source.reference)

    #########################################################
    # Load correspondence from cache if possible
    mapping = get_correspondence(original_source, original_target, corr_markers, plot=False)

    transf = Transformation(original_source, original_target, mapping)
    result = transf(original_pose)

    plt.MeshPlots.plot_correspondence(original_source, original_target, mapping).show(renderer="browser")
    plt_res.plot(original_pose, result).show(renderer="browser")
