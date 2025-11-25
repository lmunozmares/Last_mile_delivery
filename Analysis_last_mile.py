# Analysis_last_mile.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from Dataset import get_train_test_data  # matches Dataset.py (capital D)

import osmnx as ox
import networkx as nx


# -----------------------------------------------------------
# 1. Train the prediction model
# -----------------------------------------------------------
def train_model(area: str | None = "Metropolitan"):
    """
    Train a Random Forest model to predict delivery time.

    If area is not None, Dataset.get_train_test_data will try to filter by Area.
    Returns:
      model, X_test, y_test, coords_test, baseline_mae, rf_mae
    """
    if area is None:
        X_train, X_test, y_train, y_test, coords_train, coords_test = get_train_test_data()
        print("\n[train_model] Using ALL areas (no area_filter).")
    else:
        (
            X_train,
            X_test,
            y_train,
            y_test,
            coords_train,
            coords_test,
        ) = get_train_test_data(area_filter=area)
        print(f"\n[train_model] Using only Area == '{area}' when possible.")

    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=15,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    # RF predictions
    y_pred = model.predict(X_test)
    rf_mae = mean_absolute_error(y_test, y_pred)
    rf_r2 = r2_score(y_test, y_pred)

    # Naive baseline: always predict mean of y_train
    baseline_pred = np.full_like(y_test, fill_value=y_train.mean(), dtype=float)
    baseline_mae = mean_absolute_error(y_test, baseline_pred)
    baseline_r2 = r2_score(y_test, baseline_pred)

    print(f"\n[train_model] Test MAE (RF):       {rf_mae:.2f} minutes")
    print(f"[train_model] Test R² (RF):        {rf_r2:.3f}")
    print(f"[train_model] Baseline MAE (mean): {baseline_mae:.2f} minutes")
    print(f"[train_model] Baseline R² (mean):  {baseline_r2:.3f}")

    print("\n[train_model] Sample predictions vs actual (first 5):")
    for yp, yt in list(zip(y_pred, y_test))[:5]:
        print(f"  predicted={yp:.1f}  |  actual={yt:.1f}")

    return model, X_test, y_test, coords_test, baseline_mae, rf_mae


# -----------------------------------------------------------
# 2. Simulate 50 deliveries and compute time/fuel improvements
# -----------------------------------------------------------
def simulate_50_deliveries(model, X_test, y_test, coords_test, random_state: int = 0):
    """
    Simulate 50 deliveries (one 'route'):

      - Baseline: actual average delivery time (from dataset).
      - Optimized: average predicted time (if we used the model to plan).

    For simplicity, fuel usage is assumed proportional to delivery time.
    """
    rng = np.random.RandomState(random_state)

    n = len(X_test)
    if n < 50:
        raise ValueError("Not enough test samples to simulate 50 deliveries.")

    sample_indices = rng.choice(n, size=50, replace=False)

    X_sample = X_test.iloc[sample_indices]
    y_true_sample = y_test.iloc[sample_indices]
    coords_sample = (
        coords_test.iloc[sample_indices] if coords_test is not None else None
    )

    # Baseline: what actually happened
    baseline_avg_time = y_true_sample.mean()

    # Optimized: what our model predicts (as if we had planned better)
    y_pred_sample = model.predict(X_sample)
    optimized_avg_time = y_pred_sample.mean()

    # Fuel proxy: proportional to time
    baseline_fuel = baseline_avg_time
    optimized_fuel = optimized_avg_time

    improvement_pct = (
        (baseline_avg_time - optimized_avg_time) / baseline_avg_time * 100
        if baseline_avg_time > 0
        else 0.0
    )

    results = {
        "baseline_avg_time": baseline_avg_time,
        "optimized_avg_time": optimized_avg_time,
        "baseline_fuel": baseline_fuel,
        "optimized_fuel": optimized_fuel,
        "improvement_pct": improvement_pct,
        "coords_sample": coords_sample,
    }

    return results


# -----------------------------------------------------------
# 3. Simple matplotlib dashboard
# -----------------------------------------------------------
def plot_dashboard(results):
    """
    Simple dashboard: bar charts for time and fuel.
    """
    labels = ["Baseline", "Optimized"]

    times = [results["baseline_avg_time"], results["optimized_avg_time"]]
    fuels = [results["baseline_fuel"], results["optimized_fuel"]]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Delivery time chart
    axes[0].bar(labels, times)
    axes[0].set_title("Average Delivery Time (minutes)")
    axes[0].set_ylabel("Minutes")

    # Fuel usage chart
    axes[1].bar(labels, fuels)
    axes[1].set_title("Fuel Usage (relative)")
    axes[1].set_ylabel("Units")

    plt.suptitle("Last-Mile Delivery Optimization Prototype")
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------
# 4. Region selection + OSM routing
# -----------------------------------------------------------
def pick_main_region(coords_df, precision: int = 2):
    """
    Pick the geographic 'region' with the most points, based on rounded store coordinates.

    precision=2 means rounding to 0.01 degree (~1 km).
    Returns (center_lat, center_lon) for the densest region.
    """
    if coords_df is None or coords_df.empty:
        raise ValueError("coords_df is empty; cannot pick a main region.")

    lat_col = "Store_Latitude"
    lon_col = "Store_Longitude"
    if lat_col not in coords_df.columns or lon_col not in coords_df.columns:
        raise KeyError("Store_Latitude / Store_Longitude not found in coords_df.")

    rounded = coords_df[[lat_col, lon_col]].round(precision)

    counts = (
        rounded
        .value_counts()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    top = counts.iloc[0]
    center_lat = top[lat_col]
    center_lon = top[lon_col]
    print(
        f"\n[region] Densest region center (rounded): "
        f"lat={center_lat}, lon={center_lon}, count={top['count']}"
    )
    return float(center_lat), float(center_lon)


def build_osm_graph(center_lat: float, center_lon: float, dist: int = 5000):
    """
    Download the drivable road network around a point from OpenStreetMap.

    center_lat, center_lon: center of the region (degrees).
    dist: radius in meters (default 5 km).
    """
    print(
        f"\n[OSM] Downloading road network around "
        f"lat={center_lat}, lon={center_lon}, dist={dist}m ..."
    )
    G = ox.graph_from_point(
        (center_lat, center_lon),
        dist=dist,
        network_type="drive",
    )
    # Add speed (km/h) and travel_time (seconds) estimates to edges
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)
    return G


def coords_to_nodes(G, coords_df):
    """
    Map each delivery's latitude/longitude to the nearest OSM street node.
    Uses Drop_Latitude and Drop_Longitude from the Kaggle dataset.
    """
    required_cols = [
        "Drop_Latitude",
        "Drop_Longitude",
    ]
    for c in required_cols:
        if c not in coords_df.columns:
            raise KeyError(
                f"Column '{c}' not found in coords. "
                "Check your CSV and update column names."
            )

    lats = coords_df["Drop_Latitude"].values
    lons = coords_df["Drop_Longitude"].values

    # OSMnx helper: find nearest street nodes for each lat/lon pair
    nodes = ox.distance.nearest_nodes(G, X=lons, Y=lats)
    return nodes


def route_distance_and_time(G, node_order):
    """
    Compute total route distance (meters) and time (seconds)
    for a sequence of nodes, using OSM shortest paths.

    Uses an undirected view of the graph and skips legs with no path.
    """
    Gu = G.to_undirected()

    total_len = 0.0
    total_time = 0.0

    for u, v in zip(node_order[:-1], node_order[1:]):
        try:
            path = nx.shortest_path(Gu, u, v, weight="travel_time")
        except nx.NetworkXNoPath:
            # skip unreachable leg
            continue

        for n1, n2 in zip(path[:-1], path[1:]):
            edge_data = Gu[n1][n2]
            # edge_data may be a dict of edges (MultiGraph) or a single dict
            if isinstance(edge_data, dict):
                best_edge_data = min(
                    edge_data.values(),
                    key=lambda d: d.get("travel_time", float("inf")),
                )
            else:
                best_edge_data = edge_data

            total_len += best_edge_data.get("length", 0.0)
            total_time += best_edge_data.get("travel_time", 0.0)

    return total_len, total_time


def nearest_neighbor_route(G, depot_node, stop_nodes):
    """
    Very simple route optimizer:
    greedy nearest-neighbor heuristic using travel_time on the road network.

    Uses an undirected graph and ignores unreachable stops.
    """
    Gu = G.to_undirected()

    unvisited = list(stop_nodes)
    route = [depot_node]
    current = depot_node

    while unvisited:
        best_time = float("inf")
        best_idx = None

        for i, node in enumerate(unvisited):
            try:
                t = nx.shortest_path_length(Gu, current, node, weight="travel_time")
            except nx.NetworkXNoPath:
                continue

            if t < best_time:
                best_time = t
                best_idx = i

        if best_idx is None or best_time == float("inf"):
            break

        current = unvisited.pop(best_idx)
        route.append(current)

    route.append(depot_node)
    return route


def compare_routes_with_osm(coords_sample):
    """
    Use OpenStreetMap to compare baseline vs optimized route distance/time
    for deliveries in the densest region.
    """
    if coords_sample is None:
        raise ValueError("No coordinates available for routing.")

    # 1) Pick main region center from coords_sample
    center_lat, center_lon = pick_main_region(coords_sample, precision=2)

    # 2) Build road network around that center
    G = build_osm_graph(center_lat, center_lon, dist=5000)

    # 3) Nodes for the delivery points (drop locations)
    stop_nodes = coords_to_nodes(G, coords_sample)

    # 4) Define a 'depot' as the average store location in this sample
    dep_lat_col = "Store_Latitude"
    dep_lon_col = "Store_Longitude"

    if dep_lat_col not in coords_sample.columns or dep_lon_col not in coords_sample.columns:
        raise KeyError(
            "Store latitude/longitude columns not found in coords_sample. "
            "Update dep_lat_col/dep_lon_col to match your dataset."
        )

    depot_lat = coords_sample[dep_lat_col].mean()
    depot_lon = coords_sample[dep_lon_col].mean()
    depot_node = ox.distance.nearest_nodes(G, X=[depot_lon], Y=[depot_lat])[0]

    # 5) Restrict stops to same connected component as depot
    Gu = G.to_undirected()
    reachable = nx.node_connected_component(Gu, depot_node)
    filtered_stops = [n for n in stop_nodes if n in reachable]

    if len(filtered_stops) == 0:
        print("\n[OSM] No delivery stops in the same connected component as the depot.")
        print("[OSM] Skipping route comparison.")
        return None

    # 6) Baseline: depot -> stops in original order -> depot
    baseline_route_nodes = [depot_node] + list(filtered_stops) + [depot_node]
    baseline_dist, baseline_time = route_distance_and_time(G, baseline_route_nodes)

    if baseline_dist == 0 or baseline_time == 0:
        print("\n[OSM] Warning: no valid path segments accumulated for baseline route.")
        print("[OSM] Route-time improvement cannot be computed for this sample.")
        return None

    # 7) Optimized: nearest-neighbor order
    optimized_route_nodes = nearest_neighbor_route(G, depot_node, list(filtered_stops))
    optimized_dist, optimized_time = route_distance_and_time(G, optimized_route_nodes)

    if optimized_dist == 0 or optimized_time == 0:
        print("\n[OSM] Warning: no valid path segments accumulated for optimized route.")
        print("[OSM] Route-time improvement cannot be computed for this sample.")
        return None

    osm_results = {
        "baseline_km": baseline_dist / 1000,
        "optimized_km": optimized_dist / 1000,
        "baseline_minutes": baseline_time / 60,
        "optimized_minutes": optimized_time / 60,
        "improvement_pct": (baseline_time - optimized_time) / baseline_time * 100,
    }

    print("\n[OSM] Route comparison (using OpenStreetMap, densest region):")
    print(f"  Baseline distance:  {osm_results['baseline_km']:.2f} km")
    print(f"  Optimized distance: {osm_results['optimized_km']:.2f} km")
    print(f"  Baseline time:      {osm_results['baseline_minutes']:.1f} minutes")
    print(f"  Optimized time:     {osm_results['optimized_minutes']:.1f} minutes")
    print(f"  Time improvement:   {osm_results['improvement_pct']:.1f}%")

    return osm_results


# -----------------------------------------------------------
# 5. Main script tying everything together
# -----------------------------------------------------------
def main():
    # 1. Train model on one area (your "city")
    area = "Metropolitan"  # or "Urban", etc.
    model, X_test, y_test, coords_test, baseline_mae, rf_mae = train_model(area=area)

    # 2. Compute improvement based on MAE reduction (main KPI)
    mae_improvement_pct = (baseline_mae - rf_mae) / baseline_mae * 100
    print("\n=== Model Performance Improvement (Prediction Quality) ===")
    print(f"Baseline MAE: {baseline_mae:.2f} minutes")
    print(f"RF MAE:       {rf_mae:.2f} minutes")
    print(f"MAE improvement: {mae_improvement_pct:.1f}%")

    # 3. Simulate 50 deliveries and compute time/fuel improvement
    results = simulate_50_deliveries(model, X_test, y_test, coords_test, random_state=1)

    print("\n=== Simulation Results (50 Deliveries, Mean Times) ===")
    print(f"Baseline avg time (actual):  {results['baseline_avg_time']:.2f} minutes")
    print(f"Optimized avg time (pred):   {results['optimized_avg_time']:.2f} minutes")
    print(f"Time-based 'improvement':    {results['improvement_pct']:.1f}% "
          "(mean(actual) vs mean(predicted) — mainly diagnostic)")

    # 4. Show dashboard for the 50 deliveries
    plot_dashboard(results)

    # 5. Route optimization using OSM, based on the same 50 deliveries
    if results["coords_sample"] is not None:
        try:
            osm_results = compare_routes_with_osm(results["coords_sample"])
            if osm_results is None:
                print("\n[OSM] Route comparison not available for this sample.")
        except Exception as e:
            print("\n[OSM] Routing failed:", repr(e))
            print("[OSM] This part depends on osmnx, network access, and valid coordinates.")
    else:
        print("\n[OSM] Skipping routing: no latitude/longitude columns found.")


if __name__ == "__main__":
    main()
