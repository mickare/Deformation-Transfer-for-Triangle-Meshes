import os
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import yaml


class ModelConfig:
    def __init__(self, cfg: Dict[str, Any], path: Optional[str] = None):
        assert "reference" in cfg and isinstance(cfg["reference"], str)
        self.reference = cfg["reference"]
        poses = cfg.get("poses", None) or []
        self.poses: List[str] = [str(p) for p in poses]
        if path:
            self.reference = os.path.join(path, self.reference)
            self.poses = [os.path.join(path, p) for p in self.poses]


class ConfigFile:
    def __init__(self, file: str, cfg: Dict[str, Any]) -> None:
        assert "source" in cfg and isinstance(cfg["source"], dict)
        assert "target" in cfg and isinstance(cfg["target"], dict)
        self.file = file
        path = os.path.dirname(file)
        self.source = ModelConfig(cfg["source"], path)
        self.target = ModelConfig(cfg["target"], path)
        self.markers = np.array(self._load_markers(cfg.get("markers", None)))

    @classmethod
    def _load_markers(cls, markers) -> List[Tuple[int, int]]:
        if not markers:
            return []
        elif isinstance(markers, dict):
            return [(int(s), int(t)) for s, t in markers.items()]
        elif isinstance(markers, (list, tuple)):
            result: List[Tuple[int, int]] = []
            for e in markers:
                if isinstance(e, str):
                    s, t = e.split(":", maxsplit=1)
                    result.append((int(s), int(t)))
                else:
                    assert len(e) == 2
                    result.append((int(e[0]), int(e[1])))
            return result
        else:
            raise ValueError(f"invalid marker format: {type(markers)}")

    @classmethod
    def load(cls, file: str):
        with open(file, mode='rt') as fp:
            return cls(file, cfg=yaml.safe_load(fp))


def get_markers(file: str = "models/lowpoly/markers.txt"):
    markers = []
    with open(file, 'rt') as f:
        for line in f:
            if line[0] == "#":
                continue
            m = line.split(' ')
            markers.append((int(m[0]), int(m[1])))
    return np.array(markers)


config_default = ConfigFile.load("models/lowpoly/markers-cat-dog.yml")
source_reference = config_default.source.reference
target_reference = config_default.target.reference
markers = config_default.markers
