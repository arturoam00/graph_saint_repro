import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, DefaultDict, Literal

import numpy as np
import scipy.sparse as sp
import torch

from src.samplers import GraphSampler, SamplerFactory
from src.utils import adj_norm, coo_spy2torch


@dataclass
class MiniBatch:
    nodes: np.ndarray
    adj_norm: torch.Tensor
    norm_loss: torch.Tensor


class MiniBatchManager:

    adj_full_norm: torch.Tensor
    adj_train: sp.csr_matrix
    deg_train: np.ndarray
    nodes_train: np.ndarray
    nodes_val: np.ndarray
    nodes_test: np.ndarray
    n_nodes: int
    sg_remaining: DefaultDict[str, list[np.ndarray]]
    norm_factors: dict[str, np.ndarray | torch.Tensor]
    sampler: GraphSampler | None
    logger: logging.Logger

    _params: dict[str, Any]
    _is_sampler_init: bool

    def __init__(
        self,
        adj_full: sp.csr_matrix,
        adj_train: sp.csr_matrix,
        role: dict[str, list[int]],
        **kwargs: dict[str, Any],
    ) -> None:
        # Set device and additional params
        self._params = kwargs
        self._params["device"] = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Move (normalized) full adj matrix to gpu/cpu
        self.adj_full_norm = coo_spy2torch(adj_norm(adj_full).tocoo()).to(
            self._params["device"]
        )
        self.adj_train = adj_train

        # Degree diagonal for `adj_train`
        self.deg_train = np.array(self.adj_train.sum(1)).flatten()

        # Extract node ids for training, validation and testing
        self.nodes_train = np.array(role["tr"])
        self.nodes_val = np.array(role["va"])
        self.nodes_test = np.array(role["te"])

        self.n_nodes = (
            len(self.nodes_train) + len(self.nodes_val) + len(self.nodes_test)
        )

        assert (
            len(self.adj_full_norm) == self.n_nodes
        ), f"Total # of nodes should coincide with number of rows of the adj matrix"

        ## Containers for subgraph related data and normalization factors (alpha and lambda)

        # `self.sg_remaining` contains a **list of `np.array`s** for each of the possible keys:
        #   - 'indptr', 'indices', 'data', 'nodes', 'edge_index'
        # Each `np.array` in the lists corresponds to one sampled subgraph.
        self.sg_remaining = defaultdict(list)

        # `self.norm_factors` contains a `np.array` for each of the keys:
        #   - 'loss_tr`, `aggr_tr` and `loss_te`
        # Each `np.array` contains the normalization factors for each node in the full graph
        self.norm_factors = {}

        self.sampler = None
        self._is_sampler_init = False
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @property
    def n_sampled_nodes(self) -> int:
        return sum(len(nlist) for nlist in self.sg_remaining["nodes"])

    @property
    def n_subgraphs(self) -> int:
        return len(self.sg_remaining["nodes"])

    def _reset(self) -> None:
        # Empty subgraphs remaining data
        for sg in self.sg_remaining.values():
            sg.clear()

        # Reset normalization factors
        self.norm_factors["loss_tr"] = np.zeros(self.adj_train.shape[0])
        self.norm_factors["aggr_tr"] = np.zeros(self.adj_train.size).astype(np.float32)
        self.norm_factors["loss_te"] = (
            np.ones(self.adj_full_norm.shape[0]) * 1.0 / self.n_nodes
        )

    def _get_sample(self) -> None:
        sg_data = self.sampler()
        self.sg_remaining["indptr"].append(sg_data.indptr)
        self.sg_remaining["indices"].append(sg_data.indices)
        self.sg_remaining["data"].append(sg_data.data)
        self.sg_remaining["nodes"].append(sg_data.nodes)
        self.sg_remaining["edge_index"].append(sg_data.edge_index)

        self.logger.debug(
            f"Got sample with {sg_data.nodes.size} nodes and {sg_data.edge_index.size} edges"
        )

    def _load_samples(self) -> None:
        # Check for node budget exhaustion
        while not (
            self.n_sampled_nodes
            > self._params["sample_coverage"] * self.nodes_train.size
        ):
            self._get_sample()

    def _assert_no_leaks(self) -> None:
        assert np.all(self.norm_factors["loss_tr"][self.nodes_val] == 0) and np.all(
            self.norm_factors["loss_tr"][self.nodes_test] == 0
        )

    def _compute_norm_factor_aggr(self, val_max: float, val_nan: float) -> np.ndarray:
        norm_factors_aggr_tr = np.zeros_like(self.norm_factors["aggr_tr"])

        for v in range(self.adj_train.shape[0]):
            s, e = self.adj_train.indptr[v], self.adj_train.indptr[v + 1]
            val = np.clip(
                self.norm_factors["loss_tr"][v]
                / np.clip(self.norm_factors["aggr_tr"][s:e], 1e-3, None),
                0,
                val_max,
            )
            np.nan_to_num(val, nan=val_nan, copy=False)

            norm_factors_aggr_tr[s:e] = val

        return norm_factors_aggr_tr

    def _compute_norm_factor_loss(self, val_min: float) -> np.ndarray:
        self.norm_factors["loss_tr"][
            self.nodes_train[
                np.where(self.norm_factors["loss_tr"][self.nodes_train] == 0)[0]
            ]
        ] = val_min

        assert np.all(self.norm_factors["loss_tr"][self.nodes_train] >= val_min)

        # Update the loss norm factor according to the count
        # TODO: UNDERSTAND THIS LINE
        self.norm_factors["loss_tr"][self.nodes_train] = (
            self.n_subgraphs
            / self.norm_factors["loss_tr"][self.nodes_train]
            / self.nodes_train.size
        )

        return self.norm_factors["loss_tr"]

    def _compute_norm_factors(self) -> dict[str, torch.Tensor]:
        # Loop over all loaded subgraphs and update normalization factors
        # for nodes and edges. This are just counters for now
        np.add.at(
            self.norm_factors["loss_tr"], np.concatenate(self.sg_remaining["nodes"]), 1
        )
        np.add.at(
            self.norm_factors["aggr_tr"],
            np.concatenate(self.sg_remaining["edge_index"]),
            1,
        )

        # This sets the (reciprocal of) the aggregator normalization constant
        MAX_VAL, VAL_IF_NAN = 1e4, 0.1
        self.norm_factors["aggr_tr"] = self._compute_norm_factor_aggr(
            MAX_VAL, VAL_IF_NAN
        )

        # This sets the train loss norm factor
        MIN_VAL = 0.1
        self.norm_factors["loss_tr"] = self._compute_norm_factor_loss(MIN_VAL)

        self._assert_no_leaks()

        # Send loss norm factors to appropiate device
        for l in ("loss_tr", "loss_te"):
            self.norm_factors[l] = torch.from_numpy(
                self.norm_factors[l].astype(np.float32)
            ).to(self._params["device"])

        return self.norm_factors

    def _pop_norm_subgraph(self) -> tuple[np.ndarray, torch.Tensor]:
        # Make sure there's at least one sampled graph
        if self.n_subgraphs == 0:
            self._get_sample()

        nodes = self.sg_remaining["nodes"].pop()
        sg_size = len(nodes)
        adj = sp.csr_matrix(
            (
                self.sg_remaining["data"].pop(),
                self.sg_remaining["indices"].pop(),
                self.sg_remaining["indptr"].pop(),
            ),
            shape=(sg_size, sg_size),
        )
        edge_index = self.sg_remaining["edge_index"].pop()
        adj.data[:] = self.norm_factors["aggr_tr"][edge_index][:]

        # Normalize adjacency matrix according to `method`
        adj = adj_norm(adj, deg=self.deg_train, method=self._params["adj_norm"])

        # Send adj matrix to `self._params["device"]`
        adj = coo_spy2torch(adj.tocoo()).to(self._params["device"])

        return nodes, adj

    def init_sampler(self, sampler: str, **kwds) -> None:
        # Reset data
        self._reset()

        # Set sampler type
        kwds.update(self._params)
        self.sampler = SamplerFactory.get_sampler(
            sampler, self.adj_train, self.nodes_train, **kwds
        )

        s = time.time()
        self.logger.info("Running warming-up phase to compute norm factors ...")

        self._load_samples()

        e = time.time()
        self.logger.info(f"{self.n_subgraphs} subgraphs loaded in {e-s:.4f} seconds.")
        self.logger.info(f"Computing normalization factors ...")

        self.norm_factors = self._compute_norm_factors()

        self.logger.info(
            f"Normalization factors computed in {time.time() - e:.4f} sec."
        )
        self._is_sampler_init = True

    def get_batch(
        self, mode: Literal["train", "test", "val", "valtest"] = "train"
    ) -> MiniBatch:
        if not self._is_sampler_init:
            raise ValueError("Sampler not initialized")

        match mode:
            case "test" | "val" | "valtest":
                nodes = np.arange(self.n_nodes)
                adj_norm = self.adj_full_norm
                norm_loss = self.norm_factors["loss_te"]
            case "train":
                nodes, adj_norm = self._pop_norm_subgraph()
                norm_loss = self.norm_factors["loss_tr"][nodes]
            case _:
                raise ValueError(f"Invalid mode {mode}")

        return MiniBatch(nodes=nodes, adj_norm=adj_norm, norm_loss=norm_loss)
