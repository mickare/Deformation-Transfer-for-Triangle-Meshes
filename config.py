"""
Wrapper and static configuration helper.
"""
import os
from typing import Dict, Any, List, Tuple, Optional, Iterator

import numpy as np
import yaml

import meshlib


def get_markers(file: str = "models/lowpoly/markers.txt"):
    markers = []
    with open(file, 'rt') as f:
        for line in f:
            if line[0] == "#":
                continue
            m = line.split(' ')
            markers.append((int(m[0]), int(m[1])))
    return np.array(markers)


class ModelConfig:
    """Holds the path to the model reference and poses"""

    def __init__(self, cfg: Dict[str, Any], basepath: Optional[str] = None):
        assert "reference" in cfg and isinstance(cfg["reference"], str)
        self.reference = cfg["reference"]
        poses = cfg.get("poses", None) or []
        self.poses: List[str] = [str(p) for p in poses]
        if basepath:
            self.reference = os.path.join(basepath, self.reference)
            self.poses = [os.path.join(basepath, p) for p in self.poses]

    def load_reference(self) -> meshlib.Mesh:
        return meshlib.Mesh.load(self.reference)

    def load_poses(self) -> Iterator[meshlib.Mesh]:
        for p in self.poses:
            yield meshlib.Mesh.load(p)


class ConfigFile:
    """File that configures the both source & target models and the markers"""

    def __init__(self, file: str, cfg: Dict[str, Any]) -> None:
        assert "source" in cfg and isinstance(cfg["source"], dict)
        assert "target" in cfg and isinstance(cfg["target"], dict)
        self.file = file
        basepath = os.path.dirname(file)
        self.source = ModelConfig(cfg["source"], basepath)
        self.target = ModelConfig(cfg["target"], basepath)
        self.markers = self._load_markers(cfg.get("markers", None), basepath)

    @classmethod
    def _load_markers(cls, markers, basepath: str) -> np.ndarray:
        if not markers:
            return np.array([])
        elif isinstance(markers, dict):
            return np.array([(int(s), int(t)) for s, t in markers.items()], dtype=np.int)
        elif isinstance(markers, (list, tuple)):
            result: List[Tuple[int, int]] = []
            for e in markers:
                if isinstance(e, str):
                    s, t = e.split(":", maxsplit=1)
                    result.append((int(s), int(t)))
                else:
                    assert len(e) == 2
                    result.append((int(e[0]), int(e[1])))
            return np.array(result, dtype=np.int)
        elif isinstance(markers, str) and os.path.isfile(os.path.join(basepath, markers)):
            return np.asarray(get_markers(os.path.join(basepath, markers)), dtype=np.int)
        else:
            raise ValueError(f"invalid marker format: {type(markers)}")

    @classmethod
    def load(cls, file: str):
        with open(file, mode='rt') as fp:
            return cls(file, cfg=yaml.safe_load(fp))

    class Paths:
        class lowpoly:
            catdog = "models/lowpoly/markers-cat-dog.yml"
            catvoxel = "models/lowpoly/markers-cat-voxel.yml"

        class highpoly:
            cat_lion = "models/highpoly/markers-cat-lion.yml"
            horse_camel = "models/highpoly/markers-horse-camel.yml"


config_default = ConfigFile.load(ConfigFile.Paths.lowpoly.catdog)
source_reference = config_default.source.reference
target_reference = config_default.target.reference
markers = config_default.markers
