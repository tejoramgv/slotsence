"""Aisle-network travel graph (canonical dataset).

Builds an undirected graph from location_graph.csv (nodes include locations, aisle_columns,
aisle_entrances, zone_hubs, and the depot). Distances are walking distances along the aisle
network — no straight-line-through-racks. Routing uses shortest-path (Dijkstra) distances.
"""

from __future__ import annotations

import heapq
from pathlib import Path

import numpy as np
import pandas as pd

DEPOT = "DEPOT"


class NetworkGraph:
    def __init__(self):
        self.adj: dict[str, dict[str, float]] = {}

    def add_undirected(self, a: str, b: str, dist: float) -> None:
        self.adj.setdefault(a, {})[b] = dist
        self.adj.setdefault(b, {})[a] = dist

    @property
    def nodes(self) -> set:
        return set(self.adj.keys())

    def shortest(self, src: str, dst: str) -> float:
        """Dijkstra single-pair distance. Returns inf if unreachable."""
        _, dist = self.shortest_path(src, dst)
        return dist

    def shortest_path(self, src: str, dst: str) -> tuple[list[str], float]:
        """Dijkstra returning the path node sequence and total distance."""
        if src == dst:
            return [src], 0.0
        dist = {src: 0.0}
        prev: dict[str, str] = {}
        pq = [(0.0, src)]
        seen = set()
        while pq:
            d, u = heapq.heappop(pq)
            if u in seen:
                continue
            seen.add(u)
            if u == dst:
                path = []
                curr = dst
                while curr in prev:
                    path.append(curr)
                    curr = prev[curr]
                path.append(src)
                path.reverse()
                return path, d
            for w, ud in self.adj.get(u, {}).items():
                nd = d + ud
                if nd < dist.get(w, float("inf")):
                    dist[w] = nd
                    prev[w] = u
                    heapq.heappush(pq, (nd, w))
        return [], float("inf")

    def distances_from(self, src: str) -> dict[str, float]:
        """Dijkstra from one source to every reachable node (one pass, all targets)."""
        dist: dict[str, float] = {src: 0.0}
        pq = [(0.0, src)]
        seen = set()
        while pq:
            d, u = heapq.heappop(pq)
            if u in seen:
                continue
            seen.add(u)
            for w, ud in self.adj.get(u, {}).items():
                nd = d + ud
                if nd < dist.get(w, float("inf")):
                    dist[w] = nd
                    heapq.heappush(pq, (nd, w))
        return dist


def build_graph(graph_csv: Path | pd.DataFrame) -> NetworkGraph:
    if isinstance(graph_csv, Path):
        df = pd.read_csv(graph_csv)
    else:
        df = graph_csv
    g = NetworkGraph()
    for _, r in df.iterrows():
        g.add_undirected(str(r["from_node_id"]), str(r["to_node_id"]), float(r["distance_m"]))
    return g


def find_depot_node(graph: NetworkGraph) -> str | None:
    """The depot is the node whose id contains 'depot' (case-insensitive); sorted for determinism."""
    if "DEPOT" in graph.nodes:
        return "DEPOT"
    for n in sorted(graph.nodes):
        if n.lower() == "depot":
            return n
    for n in sorted(graph.nodes):
        if "depot" in n.lower():
            return n
    return None


def depot_distance_map(graph: NetworkGraph, depot: str) -> dict[str, float]:
    """Shortest distance from the depot to every reachable location node."""
    return graph.distances_from(depot)


def location_distance_map(graph: NetworkGraph, locations: list[str]) -> "DistanceMap":
    """All-pairs shortest-path distances among the given location nodes (symmetric).

    One Dijkstra per source instead of one per ordered pair, stored in a compact matrix.
    """
    return DistanceMap(graph, locations)


class DistanceMap:
    """Read-only pairwise distance lookup over a fixed list of location nodes."""

    def __init__(self, graph: NetworkGraph, locations: list[str]):
        self.locations = list(locations)
        self.index = {loc: i for i, loc in enumerate(self.locations)}
        n = len(self.locations)
        self.matrix = np.full((n, n), float("inf"))
        for i, src in enumerate(self.locations):
            dist = graph.distances_from(src)
            for dst, j in self.index.items():
                self.matrix[i, j] = 0.0 if src == dst else dist.get(dst, float("inf"))

    def get(self, key, default=None):
        a, b = key
        i = self.index.get(a)
        j = self.index.get(b)
        if i is None or j is None:
            return default
        return float(self.matrix[i, j])

    def __getitem__(self, key):
        value = self.get(key, float("inf"))
        if value == float("inf"):
            raise KeyError(key)
        return value