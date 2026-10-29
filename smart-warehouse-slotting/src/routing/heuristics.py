"""Routing heuristics for a multi-location pick trip starting/ending at the depot.

Uses shortest-path (Dijkstra) distances from the aisle network. Distances come from
`NetworkGraph.distance(a, b)` via a precomputed pairwise map for the involved locations.

Deterministic: input location lists are sorted before routing so output does not depend on
hash order.
"""

from __future__ import annotations

from .graph import DEPOT


def _pair(a: str, b: str):
    return (a, b) if a <= b else (b, a)


def nearest_neighbor(pick_locations: list[str], location_dist: dict, depot_dist: dict[str, float], depot: str = DEPOT) -> tuple[list[str], float]:
    """From the depot, greedily visit nearest unvisited pick; return to depot."""
    if not pick_locations:
        return [depot, depot], 0.0
    ordered = sorted(pick_locations)  # determinism
    unvisited = list(ordered)
    route: list[str] = [depot]
    current = depot
    distance = 0.0
    while unvisited:
        best = min(unvisited, key=lambda loc: _loc_dist(current, loc, location_dist, depot_dist, depot))
        distance += _loc_dist(current, best, location_dist, depot_dist, depot)
        route.append(best)
        current = best
        unvisited.remove(best)
    distance += _loc_dist(current, depot, location_dist, depot_dist, depot)
    route.append(depot)
    return route, distance


def _loc_dist(a: str, b: str, location_dist, depot_dist, depot: str) -> float:
    if a == depot or b == depot:
        key = a if a != depot else b
        return depot_dist.get(key, float("inf"))
    return location_dist.get(_pair(a, b), float("inf"))


def s_shape(pick_locations: list[str], location_dist: dict, depot_dist: dict[str, float], depot: str = DEPOT) -> tuple[list[str], float]:
    """Serpentine: sort by node id (a documented simplification of S-shape on the aisle grid)."""
    if not pick_locations:
        return [depot, depot], 0.0
    ordered = sorted(pick_locations)
    route = [depot] + ordered + [depot]
    distance = 0.0
    for i in range(len(route) - 1):
        distance += _loc_dist(route[i], route[i + 1], location_dist, depot_dist, depot)
    return route, distance


def route_legs(route: list[str], location_dist, depot_dist, depot: str = DEPOT) -> list[float]:
    """Distance of each consecutive leg of an already-ordered route."""
    return [
        _loc_dist(route[i], route[i + 1], location_dist, depot_dist, depot)
        for i in range(len(route) - 1)
    ]


ROUTING_STRATEGIES = {"nearest_neighbor": nearest_neighbor, "s_shape": s_shape}