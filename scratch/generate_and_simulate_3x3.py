import json
import heapq
import math
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path("smart-warehouse-slotting").resolve()
DATASET_DIR = Path("dataset").resolve()
PROCESSED = DATASET_DIR / "processed"
OUTPUTS = PROJECT_ROOT / "outputs"
EVAL = OUTPUTS / "evaluation"

def run():
    print("Generating 3x3 layout data...")
    outer_margin = 3.0
    zone_size = 9.6
    aisle_width = 3.0
    rack_depth = 1.0
    
    total_w = 2 * outer_margin + 3 * zone_size + 2 * aisle_width  # 40.8m
    total_h = total_w
    
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
                        
                        z_m = round((shelf_level + 0.5) * shelf_height, 3)
                        
                        if side == "South":
                            x_m = round(zx0 + (bay * 2 + bin_idx + 0.5) * bin_len, 3)
                            y_m = round(zy0 + rack_depth / 2.0, 3)
                            corr_x = x_m
                            corr_y = y_corrs[row]
                        elif side == "North":
                            x_m = round(zx0 + (bay * 2 + bin_idx + 0.5) * bin_len, 3)
                            y_m = round(zy1 - rack_depth / 2.0, 3)
                            corr_x = x_m
                            corr_y = y_corrs[row+1]
                        elif side == "West":
                            x_m = round(zx0 + rack_depth / 2.0, 3)
                            y_m = round(zy0 + (bay * 2 + bin_idx + 0.5) * bin_len, 3)
                            corr_x = x_corrs[col]
                            corr_y = y_m
                        else:  # East
                            x_m = round(zx1 - rack_depth / 2.0, 3)
                            y_m = round(zy0 + (bay * 2 + bin_idx + 0.5) * bin_len, 3)
                            corr_x = x_corrs[col+1]
                            corr_y = y_m
                            
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
    
    # Intersections
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
    
    col_nodes = {}
    for loc in locations:
        cx, cy = loc["_corr_x"], loc["_corr_y"]
        ckey = (cx, cy)
        if ckey not in col_nodes:
            cid = f"ACOL_{len(col_nodes)+1:04d}"
            col_nodes[ckey] = cid
            nodes[cid] = {"node_type": "aisle_column", "x_m": cx, "y_m": cy}
            
            connected = False
            for j, yc in enumerate(y_corrs):
                if abs(cy - yc) < 1e-4:
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
        lat_dist = round(math.sqrt((loc["x_m"] - cx)**2 + (loc["y_m"] - cy)**2), 3)
        edges.append((loc_id, cid, lat_dist))
        
    # Adjacency for Dijkstra
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
    
    # Save locations.csv (removing distance_to_dispatch_m per requirement)
    loc_df = pd.DataFrame(locations)
    # clean internal cols
    cols_to_save = [c for c in loc_df.columns if not c.startswith("_")]
    loc_clean = loc_df[cols_to_save]
    
    print(f"Writing {PROCESSED / 'locations.csv'} (columns: {loc_clean.columns.tolist()})")
    loc_clean.to_csv(PROCESSED / "locations.csv", index=False)
    if (OUTPUTS / "phase1").exists():
        loc_clean.to_csv(OUTPUTS / "phase1" / "locations.csv", index=False)
        
    # Format location_graph.csv
    graph_rows = []
    for u, v, d in edges:
        u_type = nodes[u]["node_type"]
        v_type = nodes[v]["node_type"]
        graph_rows.append({
            "from_node_id": u,
            "from_node_type": u_type,
            "to_node_id": v,
            "to_node_type": v_type,
            "distance_m": d,
            "data_source": "derived",
        })
    graph_df = pd.DataFrame(graph_rows)
    print(f"Writing {PROCESSED / 'location_graph.csv'} (rows: {len(graph_df)})")
    graph_df.to_csv(PROCESSED / "location_graph.csv", index=False)
    
    # Clean inventory.csv: ensure NO distance_to_dispatch column
    inv_df = pd.read_csv(PROCESSED / "inventory.csv")
    if "distance_to_dispatch_m" in inv_df.columns or "distance_to_dispatch" in inv_df.columns:
        inv_df = inv_df.drop(columns=[c for c in inv_df.columns if "distance_to_dispatch" in c])
        inv_df.to_csv(PROCESSED / "inventory.csv", index=False)
        print("Removed distance to dispatch from inventory.csv")
        
    print("Simulating routes for all batches on the 3x3 layout...")
    # Now simulate routes for baseline and optimized layouts
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.routing.graph import NetworkGraph, build_graph
    from src.routing.heuristics import nearest_neighbor, route_legs
    from src.simulation.simulator import build_batches, SimulationConfig
    
    graph = build_graph(PROCESSED / "location_graph.csv")
    
    order_items = pd.read_csv(PROCESSED / "order_items.csv")
    orders = pd.read_csv(PROCESSED / "orders.csv")
    sc_orders = set(orders[orders["scenario_id"] == "A"]["order_id"])
    order_items_sc = order_items[order_items["order_id"].isin(sc_orders)].reset_index(drop=True)
    
    cfg = SimulationConfig(routing_strategy="nearest_neighbor")
    batches = build_batches(order_items_sc, cfg.batch_size)
    print(f"Total batches to simulate: {len(batches)}")
    
    # Load slottings
    baseline_df = pd.read_csv(PROCESSED / "storage_assignments_baseline.csv")
    opt_df = pd.read_csv(OUTPUTS / "optimized" / "optimized_assignments.csv")
    
    base_slot = {}
    for r in baseline_df.itertuples(index=False):
        base_slot.setdefault(r.sku_id, []).append(r.location_id)
        
    opt_slot = {}
    for r in opt_df.itertuples(index=False):
        opt_slot.setdefault(r.sku_id, []).append(r.location_id)
        
    # Precompute pairwise distances among used locations
    used_locs = sorted(set(baseline_df["location_id"]).union(set(opt_df["location_id"])))
    print(f"Precomputing pairwise distances for {len(used_locs)} bins using graph...")
    
    # Dijkstra distances from each used location
    loc_pair_dist = {}
    for i, loc in enumerate(used_locs):
        dist_map = graph.distances_from(loc)
        for target in used_locs:
            if target in dist_map:
                loc_pair_dist[(loc, target)] = dist_map[target]
                
    class PairDistLookup:
        def get(self, pair, default=None):
            a, b = pair
            if a == b: return 0.0
            return loc_pair_dist.get((a, b), loc_pair_dist.get((b, a), default))
            
    dist_lookup = PairDistLookup()
    
    # Run simulation for baseline and optimized
    for layout_name, slotting, out_stops_file in [
        ("baseline", base_slot, EVAL / "pick_route_stops_baseline.csv"),
        ("optimized", opt_slot, EVAL / "pick_route_stops_optimized.csv")
    ]:
        print(f"Simulating layout: {layout_name}...")
        route_records = []
        stop_records = []
        
        for bi, batch in enumerate(batches):
            skus = sorted(set(batch["sku_id"]))
            pick_locations = []
            for s in skus:
                for l in slotting.get(s, []):
                    if l not in pick_locations:
                        pick_locations.append(l)
                        
            route, distance = nearest_neighbor(pick_locations, dist_lookup, depot_dists, "DEPOT")
            legs = route_legs(route, dist_lookup, depot_dists, "DEPOT")
            
            travel_time = distance / cfg.picker_speed_mps
            pick_time = len(batch) * cfg.pick_time_sec
            cycle_time = travel_time + pick_time + cfg.setup_time_sec
            
            route_records.append({
                "batch_id": bi,
                "n_orders": len(batch["order_id"].unique()),
                "n_locations": len(pick_locations),
                "distance_m": round(distance, 1),
                "travel_time_sec": round(travel_time, 1),
                "pick_time_sec": round(pick_time, 1),
                "cycle_time_sec": round(cycle_time, 1),
                "layout": layout_name,
            })
            
            batch_skus = set(batch["sku_id"])
            cum = 0.0
            for seq, (node, leg) in enumerate(zip(route, (0.0, *legs))):
                cum += leg
                if node == "DEPOT":
                    node_skus = ""
                else:
                    node_skus = ";".join(s for s in sorted(batch_skus) if node in slotting.get(s, []))
                stop_records.append({
                    "batch_id": bi,
                    "layout": layout_name,
                    "sequence": seq,
                    "location_id": node,
                    "skus": node_skus,
                    "leg_distance_m": round(leg, 3),
                    "cumulative_distance_m": round(cum, 3),
                })
                
        if layout_name == "baseline":
            base_routes_df = pd.DataFrame(route_records)
            pd.DataFrame(stop_records).to_csv(out_stops_file, index=False)
        else:
            opt_routes_df = pd.DataFrame(route_records)
            pd.DataFrame(stop_records).to_csv(out_stops_file, index=False)
            
    combined_routes = pd.concat([base_routes_df, opt_routes_df], ignore_index=True)
    combined_routes.to_csv(EVAL / "pick_routes.csv", index=False)
    print(f"Updated pick_routes.csv with {len(combined_routes)} records!")
    print(f"Baseline mean distance: {base_routes_df['distance_m'].mean():.1f}m")
    print(f"Optimized mean distance: {opt_routes_df['distance_m'].mean():.1f}m")
    
    # Update evaluation summary and metrics
    eval_summary_path = EVAL / "evaluation_summary.json"
    if eval_summary_path.exists():
        es = json.loads(eval_summary_path.read_text(encoding="utf-8"))
        # update route metrics
        for l_name, rdf in [("baseline", base_routes_df), ("optimized", opt_routes_df)]:
            if l_name in es and "metrics" in es[l_name]:
                es[l_name]["metrics"]["total_travel_distance_m"] = round(float(rdf["distance_m"].sum()), 1)
                es[l_name]["metrics"]["avg_travel_distance_m"] = round(float(rdf["distance_m"].mean()), 1)
                es[l_name]["metrics"]["total_route_time_sec"] = round(float(rdf["cycle_time_sec"].sum()), 1)
                es[l_name]["metrics"]["avg_route_time_sec"] = round(float(rdf["cycle_time_sec"].mean()), 1)
        eval_summary_path.write_text(json.dumps(es, indent=2), encoding="utf-8")
        print("Updated evaluation_summary.json")

    print("Generation and batch route update complete!")

if __name__ == "__main__":
    run()
