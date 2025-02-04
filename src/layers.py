import logging
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

aggregation_methods = {
    "mean": lambda feats: torch.stack(feats, dim=0).mean(dim=0),
    "concat": lambda feats: torch.cat(feats, dim=1),
}
activation = {"relu": F.relu, "I": lambda x: x}


class HOA(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, **kwds: dict[str, Any]) -> None:
        super().__init__()

        self._params = kwds

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        assert self._params["bias"] in ["bias", "norm", "norm-nn"], "Invalid bias type"

        self.dropout = nn.Dropout(p=self._params["dropout"])

        self.f_lin = self._initialize_linear_layers(dim_in, dim_out)
        self.f_bias, self.offset, self.scale = self._initialize_parameters(dim_out)

        self.f_norm = None
        if self._params["bias"] == "norm-nn":
            self.f_norm = self._initialize_batch_norm(dim_out)

    def _initialize_batch_norm(self, dim_out: int) -> nn.BatchNorm1d:
        """Initializes batch normalization when using 'norm-nn' bias."""
        final_dim_out = dim_out * (
            (self._params["aggr"] == "concat") * (self._params["order"] + 1)
            + (self._params["aggr"] == "mean")
        )
        return nn.BatchNorm1d(final_dim_out, eps=1e-9, track_running_stats=True)

    def _initialize_parameters(
        self, dim_out: int
    ) -> tuple[nn.ParameterList, nn.ParameterList | None, nn.ParameterList | None]:
        """Initializes bias, offset, and scale parameters."""
        f_bias = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(dim_out))
                for _ in range(self._params["order"] + 1)
            ]
        )
        offset, scale = None, None

        if self._params["bias"] in ["norm", "norm-nn"]:
            offset = nn.ParameterList(
                [
                    nn.Parameter(torch.zeros(dim_out))
                    for _ in range(self._params["order"] + 1)
                ]
            )
            scale = nn.ParameterList(
                [
                    nn.Parameter(torch.ones(dim_out))
                    for _ in range(self._params["order"] + 1)
                ]
            )

        return f_bias, offset, scale

    def _initialize_linear_layers(self, dim_in, dim_out) -> nn.ModuleList:
        return nn.ModuleList(
            [
                nn.Linear(dim_in, dim_out, bias=False)
                for _ in range(self._params["order"] + 1)
            ]
        )

    def _f_feat_trans(self, _feat: torch.Tensor, _id: int) -> torch.Tensor:
        feat = activation[self._params["act"]](
            self.f_lin[_id](_feat) + self.f_bias[_id]
        )
        if self._params["bias"] == "norm":
            mean = feat.mean(dim=1, keepdim=True)
            var = feat.var(dim=1, unbiased=False, keepdim=True) + 1e-9
            return (feat - mean) * self.scale[_id] * torch.rsqrt(var) + self.offset[_id]
        return feat

    def forward(
        self, inputs: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        adj_norm, feat_in = inputs
        feat_in = self.dropout(feat_in)
        feat_hop = [feat_in]

        assert not torch.isnan(feat_in).any()

        for _ in range(self._params["order"]):
            feat_hop.append(torch.spmm(adj_norm, feat_hop[-1]))

        feat_partial = [self._f_feat_trans(ft, idf) for idf, ft in enumerate(feat_hop)]

        assert self._params["aggr"] in aggregation_methods

        feat_out = aggregation_methods[self._params["aggr"]](feat_partial)

        if self._params["bias"] == "norm-nn":
            feat_out = self.f_norm(feat_out)

        return adj_norm, feat_out
