from dataclasses import dataclass, field
from typing import Iterable, Literal

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score


def mean(x: Iterable) -> float:
    return sum(x) / len(x)


@dataclass
class MetricsContainer:
    _loss: list[float] = field(default_factory=list)
    _f1mic: list[float] = field(default_factory=list)
    _f1mac: list[float] = field(default_factory=list)

    @property
    def loss(self) -> float:
        return mean(self._loss)

    @property
    def f1mic(self) -> float:
        return mean(self._f1mic)

    @property
    def f1mac(self) -> float:
        return mean(self._f1mac)

    def append(self, loss: float, f1mic: float, f1mac: float) -> None:
        self._loss.append(loss)
        self._f1mic.append(f1mic)
        self._f1mac.append(f1mac)


def calc_f1(y_true: torch.Tensor, y_pred: torch.Tensor, is_sigmoid: bool) -> float:
    if is_sigmoid:
        y_pred = (F.sigmoid(y_pred) > 0.5).int()
        # y_pred = (y_pred > 0.5).int()
    else:
        y_pred = torch.argmax(F.softmax(y_pred, dim=1), dim=1)
        # y_pred = torch.argmax(y_pred, dim=1)

    # Move data to cpu
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    return f1_score(y_true, y_pred, average="micro"), f1_score(
        y_true, y_pred, average="macro"
    )


def coo_spy2torch(adj: sp.coo_matrix) -> torch.Tensor:
    values = adj.data
    coords = np.array(adj.coords)
    return torch.sparse_coo_tensor(
        indices=coords, values=values, size=adj.shape, dtype=torch.float
    )


def adj_norm(
    adj: sp.csr_matrix,
    deg: np.ndarray | None = None,
    method: Literal["rw", "cw", "sym"] = "rw",
    sort_indices: bool = True,
) -> sp.csr_matrix:
    diag_shape = adj.shape
    deg = adj.sum(1).A1 if deg is None else deg.flatten()
    deg += 1
    norm_deg_mat = sp.diags_array(1 / deg, shape=diag_shape)

    match method:
        case "rw":
            adj_norm = norm_deg_mat @ adj
        case "cw":
            adj_norm = adj @ norm_deg_mat
        case "sym":
            adj_norm = norm_deg_mat ** (1 / 2) @ adj @ norm_deg_mat ** (1 / 2)

    if sort_indices:
        adj_norm.sort_indices()

    return adj_norm
