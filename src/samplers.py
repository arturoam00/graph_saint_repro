from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp
from numpy.linalg import norm

from src.utils import adj_norm


@dataclass
class SubGraphData:
    indptr: np.ndarray
    indices: np.ndarray
    data: np.ndarray
    nodes: np.ndarray
    edge_index: np.ndarray

    def __eq__(self, value: Any) -> bool:
        if isinstance(value, SubGraphData):
            return (
                np.all(self.indptr == value.indptr)
                and np.all(self.indices == value.indices)
                and np.all(self.data == value.data)
                and np.all(self.nodes == value.nodes)
                and np.all(self.edge_index == value.edge_index)
            )
        return NotImplemented


class GraphSampler(ABC):
    adj_train: sp.csr_matrix
    node_train: np.ndarray
    _params: dict[str, Any]

    @property
    @abstractmethod
    def sg_budget(self) -> int:
        pass

    def __init__(self, adj_train: sp.csr_matrix, node_train: np.ndarray, **kwds):
        self.adj_train = adj_train
        self.node_train = np.unique(node_train).astype(np.int32)
        self._params = kwds

    def _extract_subgraph(self, nodes: np.ndarray) -> SubGraphData:
        nodes = np.sort(np.unique(nodes))
        n_nodes = nodes.size
        indptr = np.zeros(n_nodes + 1)
        indices, edge_index = [], []

        # Create a map for node ids in the extracted sg
        orig2sg = {n: i for i, n in enumerate(nodes)}

        for node in nodes:
            s, e = self.adj_train.indptr[node], self.adj_train.indptr[node + 1]
            neighs = self.adj_train.indices[s:e]

            for i, neigh in enumerate(neighs):
                if neigh in orig2sg:
                    indices.append(orig2sg[neigh])
                    indptr[orig2sg[node] + 1] += 1
                    edge_index.append(s + i)

        indptr = indptr.cumsum().astype(np.int64)
        indices = np.array(indices)
        edge_index = np.array(edge_index)
        data = np.ones(indices.size)

        assert indptr[-1] == indices.size == edge_index.size

        return SubGraphData(
            indptr=indptr,
            indices=indices,
            data=data,
            nodes=nodes,
            edge_index=edge_index,
        )

    @abstractmethod
    def _sample(self) -> np.ndarray:
        pass

    def __call__(self, *args, **kwds) -> SubGraphData:
        return self._extract_subgraph(self._sample(*args, **kwds))


class ProbabilisticGraphSampler(GraphSampler):

    def __init__(self, adj_train, node_train, **kwds):
        super().__init__(adj_train, node_train, **kwds)
        self.probs = self._probs()

    @abstractmethod
    def _probs(self) -> np.ndarray | None:
        pass

    def _sample(self):
        return np.random.choice(
            self.node_train, replace=True, size=self.sg_budget, p=self.probs
        )


class UniformRandomSampler(ProbabilisticGraphSampler):

    @property
    def sg_budget(self) -> int:
        return self._params["sg_size"]

    def _probs(self):
        return None


class NodeSampler(ProbabilisticGraphSampler):

    @property
    def sg_budget(self) -> int:
        return self._params["sg_size"]

    def _probs(self) -> np.ndarray:
        probs = np.array(
            (adj_norm(self.adj_train, method="rw") ** 2).sum(axis=0)
        ).squeeze()[self.node_train]
        return probs / np.sum(probs)


class EdgeSampler(GraphSampler):

    @property
    def sg_budget(self) -> int:
        return 2 * self._params["size_sg_edge"]

    def __init__(self, adj_train, node_train, **kwds):
        super().__init__(adj_train, node_train, **kwds)
        self.edge_probs_m = self._set_edges()
        self.edges = np.array(
            [(row, col) for row, col in zip(*self.edge_probs_m.coords)]
        )
        self.probs = self.edge_probs_m.data / self.edge_probs_m.data.sum()

    def _set_edges(self) -> sp.coo_matrix:
        adj_train_norm = adj_norm(self.adj_train, method="rw")

        # Calculate edge probabilities
        edge_probs_m = sp.csr_matrix(
            (
                np.zeros(self.adj_train.size),
                self.adj_train.indices,
                self.adj_train.indptr,
            ),
            shape=self.adj_train.shape,
        )
        edge_probs_m.data[:] = adj_train_norm.data[:]
        adj_t = adj_train_norm.tocsc()
        edge_probs_m.data += adj_t.data
        edge_probs_m.data *= self.sg_budget / edge_probs_m.data.sum()
        edge_probs_m = sp.triu(edge_probs_m).astype(np.float32)

        return edge_probs_m

    def _sample(self) -> np.ndarray:
        edges_ids = np.random.choice(
            np.arange(len(self.edges)),
            size=self.sg_budget,
            replace=True,
            p=self.probs,
        )
        return np.unique(np.array(self.edges[edges_ids]).flatten())


class BaseRWSampler(GraphSampler):

    @property
    def sg_budget(self) -> int:
        return self._params["n_roots"] * self._params["depth"]

    @abstractmethod
    def _rw(self, roots: np.ndarray) -> np.ndarray:
        pass

    def _sample(self) -> np.ndarray:
        roots = np.random.choice(
            self.node_train, size=self._params["n_roots"], replace=True
        )
        return self._rw(roots=roots)


class RWSampler(BaseRWSampler):

    def _rw(self, roots: np.ndarray) -> np.ndarray:
        sampled = set(roots)

        # For each root, perform a random walk of length h.
        for root in roots:
            u = root
            for _ in range(self._params["depth"]):
                # Get neighbors of node u (neighbors are given by the column indices in the row u of A).
                neighbors = self.adj_train[u].indices
                # If the current node has no neighbors, break out of the walk.
                if len(neighbors) == 0:
                    break
                # Sample a neighbor uniformly at random.
                u = np.random.choice(neighbors)
                # Add the new node to the set of sampled indices.
                sampled.add(u)

        return np.fromiter(sampled, dtype=np.int32)


class FeatureAwareSampler(BaseRWSampler):

    # -- This is very much the same as `RWSampler`
    #    I intentionally repeat the code to avoid function calls inside for loops
    def _rw(self, roots: np.ndarray) -> np.ndarray:
        assert self._params["similarity"] in (
            "min",
            "max",
        ), f"Invalid '{self._params["similarity"]}"

        sampled = set(roots)

        for root in roots:
            u = root
            r_feat = self._params["feats"][u]
            for _ in range(self._params["depth"]):
                neighbors = self.adj_train[u].indices
                neighbors_map = {i: n for i, n in enumerate(neighbors)}

                if len(neighbors) == 0:
                    break

                n_feats = self._params["feats"][neighbors]
                cosim = np.dot(n_feats, r_feat) / norm(n_feats, axis=1) / norm(r_feat)
                u_idx = (
                    np.argmax(cosim)
                    if self._params["similarity"] == "max"
                    else np.argmin(cosim)
                )
                sampled.add(neighbors_map[u_idx])

        return np.fromiter(sampled, dtype=np.int32)


class SamplerFactory:
    __samplers__ = {
        "uniform_rnd": UniformRandomSampler,
        "node": NodeSampler,
        "edge": EdgeSampler,
        "rw": RWSampler,
        "fa": FeatureAwareSampler,
    }

    @classmethod
    def get_sampler(
        cls, _sampler: str, _adj_train: sp.csr_matrix, _node_train: np.ndarray, **kwds
    ) -> GraphSampler:
        assert _sampler in cls.__samplers__, f"Invalid sampler name `{_sampler}`"
        return cls.__samplers__[_sampler](_adj_train, _node_train, **kwds)
