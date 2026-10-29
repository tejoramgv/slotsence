import json
import heapq
import math
from pathlib import Path
import pandas as pd
import numpy as np

def generate_3x3_layout():
    outer_margin = 3.0
    zone_size = 9.6
    aisle_width = 3.0
    rack_depth = 1.0
    
    total_w = 2 * outer_margin + 3 * zone_size + 2 * aisle_width  # 40.8m
    total_h = total_w
    
    # 9 zones
    zone_types = [
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
    
    bin_classes = {
        "small": {"dims_cm": [45.0, 30.0, 40.0], "shelf_weight_kg": [40.0, 35.0, 30.0, 25.0]},
        "standard": {"dims_cm": [60.0, 40.0, 45.0], "shelf_weight_kg": [80.0, 60.0, 50.0, 40.0]},
        "bulk": {"dims_cm": [100.0, 70.0, 70.0], "shelf_weight_kg": [200.0, 160.0, 140.0, 120.0]},
        "heavy": {"dims_cm": [60.0, 40.0, 45.0], "shelf_weight_kg": [120.0, 100.0, 80.0, 60.0]},
        "hazard": {"dims_cm": [60.0, 40.0, 45.0], "shelf_weight_kg": [60.0, 50.0, 40.0, 30.0]},
    }
    zone_rules = {
        "general": {"allowed_handling": ["standard", "fragile"], "allowed_hazard": ["NONE"]},
        "small_parts": {"allowed_handling": ["standard", "fragile"], "allowed_hazard": ["NONE"]},
        "bulk": {"allowed_handling": ["standard", "fragile"], "allowed_hazard": ["NONE"]},
        "heavy": {"allowed_handling": ["standard", "heavy"], "allowed_hazard": ["NONE"]},
        "hazard": {"allowed_handling": ["standard"], "allowed_hazard": ["flammable", "corrosive", "toxic"]},
    }
    
    zones = {}
    for idx, (zid, ztype, bclass) in enumerate(zone_types):
        col = idx % 3
        row = idx // 3
        x0 = outer_margin + col * (zone_size + aisle_width)
        y0 = outer_margin + row * (zone_size + aisle_width)
        zones[zid] = {
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
        
    x_corrs = [
        round(outer_margin / 2.0, 3),  # 1.5
        round(outer_margin + zone_size + aisle_width / 2.0, 3),  # 14.1
        round(outer_margin + 2 * zone_size + 1.5 * aisle_width, 3),  # 26.7
        round(outer_margin + 3 * zone_size + 2 * aisle_width + outer_margin / 2.0, 3),  # 39.3
    ]
    y_corrs = [
        round(outer_margin / 2.0, 3),  # 1.5
        round(outer_margin + zone_size + aisle_width / 2.0, 3),  # 14.1
        round(outer_margin + 2 * zone_size + 1.5 * aisle_width, 3),  # 26.7
        round(outer_margin + 3 * zone_size + 2 * aisle_width + outer_margin / 2.0, 3),  # 39.3
    ]
    
    locations = []
    loc_id_counter = 0
    
    # 4 sides: South (0), North (1), West (2), East (3)
    # Each zone has 4 racks on its 4 sides
    # 8 bays per rack, 4 shelves, 2 bins per shelf = 64 bins per rack * 4 racks = 256 bins per zone
    bay_len = zone_size / 8.0  # 1.2m
    bin_len = bay_len / 2.0    # 0.6m
    shelf_height = 0.5
    
    for idx, (zid, ztype, bclass) in enumerate(zone_types):
        zinfo = zones[zid]
        col = zinfo["col"]
        row = zinfo["row"]
        zx0, zy0, zx1, zy1 = zinfo["x0"], zinfo["y0"], zinfo["x1"], zinfo["y1"]
        
        bc = bin_classes[bclass]
        bin_w_cm, bin_h_cm, bin_d_cm = (float(v) for v in bc["dims_cm"])
        shelf_weights = [float(v) for v in bc["shelf_weight_kg"]]
        handling_rule = list(zone_rules[ztype]["allowed_handling"])
        hazard_rule = list(zone_rules[ztype]["allowed_hazard"])
        rack_max_weight = round(0.85 * sum(shelf_weights) * 2 * 8, 1)
        
        sides = ["South", "North", "West", "East"]
        for side_idx, side in enumerate(sides):
            rack_id = f"R{idx+1}-{side}"
            aisle_id = f"A{idx+1}-{side}"
            
            for bay in range(8):
                for shelf_level in range(4):
                    for bin_idx in range(2):
                        loc_id_counter += 1
                        loc_id = f"LOC{loc_id_counter:06d}"
                        
                        # Coordinates
                        z_m = round((shelf_level + 0.5) * shelf_height, 3)
                        
                        if side == "South":
                            # along bottom edge, faces y_corrs[row]
                            x_m = round(zx0 + (bay * 2 + bin_idx + 0.5) * bin_len, 3)
                            y_m = round(zy0 + rack_depth / 2.0, 3)
                            corr_x = x_m
                            corr_y = y_corrs[row]
                        elif side == "North":
                            # along top edge, faces y_corrs[row+1]
                            x_m = round(zx0 + (bay * 2 + bin_idx + 0.5) * bin_len, 3)
                            y_m = round(zy1 - rack_depth / 2.0, 3)
                            corr_x = x_m
                            corr_y = y_corrs[row+1]
                        elif side == "West":
                            # along left edge, faces x_corrs[col]
                            x_m = round(zx0 + rack_depth / 2.0, 3)
                            y_m = round(zy0 + (bay * 2 + bin_idx + 0.5) * bin_len, 3)
                            corr_x = x_corrs[col]
                            corr_y = y_m
                        else:  # East
                            # along right edge, faces x_corrs[col+1]
                            x_m = round(zx1 - rack_depth / 2.0, 3)
                            y_m = round(zy0 + (bay * 2 + bin_idx + 0.5) * bin_len, 3)
                            corr_x = x_corrs[col+1]
                            corr_y = y_m
                            
                        # Ergonomic zone
                        if shelf_level in (1, 2):
                            ergonomic = "golden"
                        elif shelf_level == 0:
                            ergonomic = "secondary"
                        else:
                            ergonomic = "reserve"
                            
                        allowed_handling = list(handling_rule)
                        if ztype == "heavy" and shelf_level > 2:
                            allowed_handling = [h for h in allowed_handling if h != "heavy"]
                            
                        locations.append({
                            "location_id": loc_id,
                            "warehouse_id": "WH001",
                            "zone_id": zid,
                            "zone_type": ztype,
                            "aisle_id": aisle_id,
                            "rack_id": rack_id,
                            "bay": bay,
                            "shelf_level": shelf_level,
                            "x_m": x_m,
                            "y_m": y_m,
                            "z_m": z_m,
                            "width_cm": bin_w_cm,
                            "height_cm": bin_h_cm,
                            "depth_cm": bin_d_cm,
                            "max_weight_kg": shelf_weights[shelf_level],
                            "usable_volume_cm3": round(bin_w_cm * bin_h_cm * bin_d_cm, 2),
                            "shelf_max_weight_kg": shelf_weights[shelf_level],
                            "rack_max_weight_kg": rack_max_weight,
                            "ergonomic_zone": ergonomic,
                            "allowed_handling": "|".join(allowed_handling),
                            "allowed_hazard": "|".join(hazard_rule),
                            "data_source": "synthetic",
                            "data_quality_status": "valid",
                            # internal fields
                            "_corr_x": corr_x,
                            "_corr_y": corr_y,
                            "_side": side,
                        })
                        
    # Build Graph
    nodes = {}
    edges = []
    depot_x = round(total_w / 2.0, 3)
    depot_y = 0.5
    nodes["DEPOT"] = {"node_type": "depot", "x_m": depot_x, "y_m": depot_y}
    depot_corr = "CORR_DEPOT_ENTRY"
    nodes[depot_corr] = {"node_type": "corridor_node", "x_m": depot_x, "y_m": y_corrs[0]}
    edges.append(("DEPOT", depot_corr, round(abs(y_corrs[0] - depot_y), 3)))
    
    # Corridor intersections
    for i, xc in enumerate(x_corrs):
        for j, yc in enumerate(y_corrs):
            nodes[f"INT_{i}_{j}"] = {"node_type": "intersection", "x_m": xc, "y_m": yc}
            
    # Connect intersections along corridor segments
    for i in range(len(x_corrs)):
        for j in range(len(y_corrs) - 1):
            n1 = f"INT_{i}_{j}"
            n2 = f"INT_{i}_{j+1}"
            edges.append((n1, n2, round(abs(y_corrs[j+1] - y_corrs[j]), 3)))
    for j in range(len(y_corrs)):
        for i in range(len(x_corrs) - 1):
            n1 = f"INT_{i}_{j}"
            n2 = f"INT_{i+1}_{j}"
            edges.append((n1, n2, round(abs(x_corrs[i+1] - x_corrs[i]), 3)))
            
    edges.append((depot_corr, "INT_1_0", round(abs(depot_x - x_corrs[1]), 3)))
    edges.append((depot_corr, "INT_2_0", round(abs(x_corrs[2] - depot_x), 3)))
    
    # For each location, create an aisle column node on the corridor
    # Group locations by (corr_x, corr_y) to share aisle columns
    col_nodes = {}
    for loc in locations:
        cx, cy = loc["_corr_x"], loc["_corr_y"]
        ckey = (cx, cy)
        if ckey not in col_nodes:
            cid = f"ACOL_{len(col_nodes)+1:04d}"
            col_nodes[ckey] = cid
            nodes[cid] = {"node_type": "aisle_column", "x_m": cx, "y_m": cy}
            
            # Connect column node to the corridor network:
            # Check if this column is on a horizontal corridor or vertical corridor
            # Connect it to the two nearest intersection nodes on that corridor
            connected = False
            for j, yc in enumerate(y_corrs):
                if abs(cy - yc) < 1e-4:
                    # On horizontal corridor j: find i such that x_corrs[i] <= cx <= x_corrs[i+1]
                    for i in range(len(x_corrs) - 1):
                        if x_corrs[i] <= cx <= x_corrs[i+1]:
                            edges.append((cid, f"INT_{i}_{j}", round(abs(cx - x_corrs[i]), 3)))
                            edges.append((cid, f"INT_{i+1}_{j}", round(abs(x_corrs[i+1] - cx), 3)))
                            connected = True
                            break
                    break
            if not connected:
                for i, xc in enumerate(x_corrs):
                    if abs(cx - xc) < 1e-4:
                        # On vertical corridor i: find j such that y_corrs[j] <= cy <= y_corrs[j+1]
                        for j in range(len(y_corrs) - 1):
                            if y_corrs[j] <= cy <= y_corrs[j+1]:
                                edges.append((cid, f"INT_{i}_{j}", round(abs(cy - y_corrs[j]), 3)))
                                edges.append((cid, f"INT_{i}_{j+1}", round(abs(y_corrs[j+1] - cy), 3)))
                                connected = True
                                break
                        break
                        
        cid = col_nodes[ckey]
        loc_id = loc["location_id"]
        nodes[loc_id] = {"node_type": "location", "x_m": loc["x_m"], "y_m": loc["y_m"]}
        # lateral distance from rack to corridor
        lat_dist = round(math.sqrt((loc["x_m"] - cx)**2 + (loc["y_m"] - cy)**2), 3)
        edges.append((loc_id, cid, lat_dist))
        
    print(f"Total locations: {len(locations)}")
    print(f"Total graph nodes: {len(nodes)}, edges: {len(edges)}")
    
    # Test Dijkstra connectivity
    adj = {}
    for u, v, d in edges:
        adj.setdefault(u, {})[v] = d
        adj.setdefault(v, {})[u] = d
        
    def dijkstra(src):
        dist = {src: 0.0}
        pq = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]: continue
            for v, w in adj.get(u, {}).items():
                if d + w < dist.get(v, float('inf')):
                    dist[v] = d + w
                    heapq.heappush(pq, (d + w, v))
        return dist
        
    depot_dists = dijkstra("DEPOT")
    loc_dists = [depot_dists.get(loc["location_id"], float("inf")) for loc in locations]
    print(f"Min depot dist: {min(loc_dists):.2f}m, Max depot dist: {max(loc_dists):.2f}m, Avg depot dist: {np.mean(loc_dists):.2f}m")
    assert all(d < float("inf") for d in loc_dists), "All locations must be reachable from DEPOT!"
    print("ALL 2,304 LOCATIONS ARE REACHABLE VIA INTER-ZONE CORRIDORS!")

if __name__ == "__main__":
    generate_3x3_layout()
