import logging
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.layers import HOA


class GraphSAINT(nn.Module):
    def __init__(
        self, feats: np.ndarray, labels: np.ndarray, **kwds: dict[str, Any]
    ) -> None:
        super().__init__()
        self._params = kwds
        # Use GPU when possible
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Store data as torch tensors and move to device
        self.feats = torch.from_numpy(feats.astype(np.float32)).to(self._device)
        self.labels = torch.from_numpy(labels.astype(np.float32)).to(self._device)

        # Network architecture
        self.aggr_layer = HOA
        self.conv_layers = self.set_conv_layers()
        self.classifier = self.conv_layers.pop(-1)

        if not self._params["loss"] == "sigmoid":
            self.labels = torch.argmax(self.labels, dim=1)

    def forward(
        self, adj: torch.Tensor, node_ids: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.feats[node_ids]
        labels = self.labels[node_ids]
        _, out = self.conv_layers((adj, feats))
        out = F.normalize(out, p=2, dim=1)
        _, pred = self.classifier((None, out))
        return pred, labels

    def set_conv_layers(self) -> nn.Sequential:
        layers = self._params["arch"].split("-")
        modules = nn.ModuleList()
        is_concat = self._params["aggr"] == "concat"

        # Build hidden layers
        dim_out = self._params["dim"]
        for i, l in enumerate(layers):
            dim_in = (
                self.feats.shape[1]
                if i == 0
                else (is_concat * int(layers[i - 1]) + 1) * dim_out
            )
            self._params.update({"order": int(l)})
            modules.append(self.aggr_layer(dim_in, dim_out, **self._params))

        # Classifier
        self._params.update({"order": 0, "act": "I", "bias": "bias"})
        dim_in = (is_concat * int(layers[-1]) + 1) * dim_out
        modules.append(
            HOA(dim_in=dim_in, dim_out=self._params["n_classes"], **self._params)
        )

        return nn.Sequential(*modules)
