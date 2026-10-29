import math
import heapq
import pandas as pd
import numpy as np

def build_3x3_warehouse():
    # 9 zones in 3x3 matrix
    # Warehouse box: outer_margin = 3.0, zone_size = 9.6, aisle_width = 3.0
    outer_margin = 3.0
    zone_size = 9.6
    aisle_width = 3.0
    rack_depth = 1.0
    
    total_w = 2 * outer_margin + 3 * zone_size + 2 * aisle_width # 40.8m
    total_h = total_w
    
    zones = []
    zone_classes = [
        ("Z1", "general", "standard"),
        ("Z2", "general", "standard"),
        ("Z3", "general", "standard"),
        ("Z4", "small_parts", "small"),
        ("Z5", "bulk", "bulk"),
        ("Z6", "bulk", "bulk"),
        ("Z7", "heavy", "heavy"),
        ("Z8", "heavy", "heavy"),
        ("Z9", "hazard", "hazard"),
    ]
    
    zone_dict = {}
    for idx, (zid, ztype, bclass) in enumerate(zone_classes):
        col = idx % 3
        row = idx // 3
        x0 = outer_margin + col * (zone_size + aisle_width)
        y0 = outer_margin + row * (zone_size + aisle_width)
        zone_info = {
            "zone_id": zid,
            "zone_type": ztype,
            "bin_class": bclass,
            "col": col,
            "row": row,
            "x0": round(x0, 3),
            "y0": round(y0, 3),
            "x1": round(x0 + zone_size, 3),
            "y1": round(y0 + zone_size, 3),
            "x_center": round(x0 + zone_size / 2.0, 3),
            "y_center": round(y0 + zone_size / 2.0, 3),
        }
        zones.append(zone_info)
        zone_dict[zid] = zone_info

    # 4 corridor horizontal centerlines, 4 vertical centerlines
    x_corrs = [
        round(outer_margin / 2.0, 3), # 1.5
        round(outer_margin + zone_size + aisle_width / 2.0, 3), # 14.1
        round(outer_margin + 2 * zone_size + 1.5 * aisle_width, 3), # 26.7
        round(outer_margin + 3 * zone_size + 2 * aisle_width + outer_margin / 2.0, 3), # 39.3
    ]
    y_corrs = [
        round(outer_margin / 2.0, 3), # 1.5
        round(outer_margin + zone_size + aisle_width / 2.0, 3), # 14.1
        round(outer_margin + 2 * zone_size + 1.5 * aisle_width, 3), # 26.7
        round(outer_margin + 3 * zone_size + 2 * aisle_width + outer_margin / 2.0, 3), # 39.3
    ]
    
    # Corridor intersection nodes
    nodes = {}
    edges = []
    
    # Add depot at front center entrance
    depot_x = round(total_w / 2.0, 3) # 20.4
    depot_y = 0.5
    nodes["DEPOT"] = {"node_type": "depot", "x_m": depot_x, "y_m": depot_y}
    # Connect DEPOT to bottom corridor y=1.5 at x=20.4
    depot_corr_node = "CORR_DEPOT_ENTRY"
    nodes[depot_corr_node] = {"node_type": "corridor_node", "x_m": depot_x, "y_m": y_corrs[0]}
    edges.append(("DEPOT", depot_corr_node, round(abs(y_corrs[0] - depot_y), 3)))
    
    # Intersections
    for i, xc in enumerate(x_corrs):
        for j, yc in enumerate(y_corrs):
            node_id = f"INT_{i}_{j}"
            nodes[node_id] = {"node_type": "intersection", "x_m": xc, "y_m": yc}
            
    # Connect grid of intersections
    for i, xc in enumerate(x_corrs):
        for j in range(len(y_corrs) - 1):
            n1 = f"INT_{i}_{j}"
            n2 = f"INT_{i}_{j+1}"
            dist = round(abs(y_corrs[j+1] - y_corrs[j]), 3)
            edges.append((n1, n2, dist))
            
    for j, yc in enumerate(y_corrs):
        for i in range(len(x_corrs) - 1):
            n1 = f"INT_{i}_{j}"
            n2 = f"INT_{i+1}_{j}"
            dist = round(abs(x_corrs[i+1] - x_corrs[i]), 3)
            edges.append((n1, n2, dist))
            
    # Connect depot entry to the bottom corridor intersections INT_1_0 (14.1) and INT_2_0 (26.7)
    edges.append((depot_corr_node, "INT_1_0", round(abs(depot_x - x_corrs[1]), 3)))
    edges.append((depot_corr_node, "INT_2_0", round(abs(x_corrs[2] - depot_x), 3)))
    
    print(f"Warehouse bounds: [0, {total_w}] x [0, {total_h}]")
    print(f"Corridors X: {x_corrs}, Y: {y_corrs}")
    print(f"Nodes in grid: {len(nodes)}, Edges: {len(edges)}")

if __name__ == "__main__":
    build_3x3_warehouse()
