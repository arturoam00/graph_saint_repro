import json
import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler


@dataclass
class DataContainer:
    adj_full: sp.csr_matrix
    adj_train: sp.csr_matrix
    feats: np.ndarray
    class_arr: np.ndarray
    role: dict[str, list[int]]


class DataLoader:
    adj_full: sp.csr_matrix
    adj_train: sp.csr_matrix
    role: dict[str, list[int]]
    feats: np.ndarray
    class_map: dict[int, int | list[int]]
    n_classes: int
    logger: logging.Logger
    _scaler: StandardScaler | None
    _classification: Literal["m", "s"]

    def __init__(self, prefix: str, normalize: bool = True):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Load data from the `prefix` directory
        self._load_data(prefix=prefix)

        # Validate once loaded
        self._validate_data()

        # Normalize features if needed
        self._scaler = None
        if normalize:
            self._normalize()

        self.logger.info(
            f""" -- Loaded data from `{prefix}` --
        Number of nodes: {self.adj_full.shape[0]}
        Number of edges: {len(self.adj_full.nonzero()[0])}
        Number of training edges: {len(self.adj_train.nonzero()[0])}
        Train/Val/Test: {self.tr_va_test_split()}
        Number of features: {self.feats.shape[1]}
        Number of classes: {self.n_classes}
        Type of classification: `{self._classification}`"""
        )

    def _load_data(self, prefix: str) -> None:
        self.adj_full = sp.load_npz(f"./{prefix}/adj_full.npz").astype(np.bool)
        self.adj_train = sp.load_npz(f"./{prefix}/adj_train.npz").astype(np.bool)

        with open(f"./{prefix}/role.json") as f:
            self.role = json.load(f)

        self.feats = np.load(f"./{prefix}/feats.npy")

        with open(f"./{prefix}/class_map.json") as f:
            _class_map = json.load(f)
        self.class_map = {int(i): c for i, c in _class_map.items()}

    def _normalize(self) -> None:
        train_nodes = np.unique(np.array(self.adj_train.nonzero()[0]))
        train_feats = self.feats[train_nodes]
        self._scaler = StandardScaler()
        self._scaler.fit(train_feats)
        self.feats = self._scaler.transform(self.feats)

    def _validate_data(self) -> None:
        assert len(self.class_map) == self.feats.shape[0]

        # Sparse matrices check
        _label = "ADJACENCY MATRICES"
        assert isinstance(self.adj_full, sp.csr_matrix) and isinstance(
            self.adj_train, sp.csr_matrix
        ), f"Validation of {_label} failed"

        # Role dict check
        _label = "ROLE"
        assert (
            all(isinstance(k, str) for k in self.role.keys())
            and all(isinstance(v, list) for v in self.role.values())
            and all(
                all(isinstance(i, int) for i in vals) for vals in self.role.values()
            )
        ), f"Validation of {_label} failed"

        # Class map check
        _label = "CLASS MAP"
        assert all(
            isinstance(k, int) for k in self.class_map.keys()
        ), f"Validation {_label} failed"

        if all(isinstance(v, int) for v in self.class_map.values()):
            self._classification = "s"
            self.n_classes = len(set(self.class_map.values()))
        elif all(isinstance(v, list) for v in self.class_map.values()):
            self._classification = "m"
            self.n_classes = len(list(self.class_map.values())[0])
            assert all(
                all(isinstance(i, (float, int)) for i in l)
                for l in self.class_map.values()
            ), f"Validation {_label} failed"
        else:
            raise ValueError(f"Validation {_label} failed")

    def tr_va_test_split(self) -> str:
        n_nodes = self.adj_full.shape[0]
        return "/".join(
            str(round(len(self.role[i]) / n_nodes, 2)) for i in ("tr", "va", "te")
        )

    def transform(self, feats: np.ndarray) -> np.ndarray:
        return self._scaler.transform(feats)

    def get_data(self) -> DataContainer:
        n_nodes = self.adj_full.shape[0]
        class_arr = np.zeros((n_nodes, self.n_classes))
        nodes = np.array(list(self.class_map.keys()))
        labels = np.array(list(self.class_map.values()))

        if self._classification == "s":
            offset = labels.min()
            class_arr[nodes, labels - offset] = 1
        else:
            class_arr[nodes] = labels

        return DataContainer(
            adj_full=self.adj_full,
            adj_train=self.adj_train,
            feats=self.feats,
            class_arr=class_arr.astype(np.int64),
            role=self.role,
        )

    def __call__(self, *args, **kwds):
        return self.get_data()
