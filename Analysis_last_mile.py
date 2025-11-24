# analysis_last_mile.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

import networkx as nx
import osmnx as ox  # uses OpenStreetMap for routing

from Dataset import get_train_test_data


# -----------------------------------------------------------
# 1. Train the prediction model
# -----------------------------------------------------------
def train_model(area: str = "Urban"):
    """
    Train a Random Forest model to predict delivery time for a given area.

    Returns:
      model, X_test, y_test, coords_test
    """
    (
        X_train,
        X_test,
        y_train,
        y_test,
        coords_train,
        coords_test,
    ) = get_train_test_data(area_filter=area)

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,  # fixed seed for reproducibility
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    # Evaluation on the test set
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"\n[train_model] Test MAE for area={area}: {mae:.2f} minutes")

    # Show a small sample of predictions vs actual
    print("\n[train_model] Sample predictions vs actual (first 5):")
    for yp, yt in list(zip(y_pred, y_test))[:5]:
        print(f"  predicted={yp:.1f}  |  actual={yt:.1f}")

    return model, X_test, y_test, coords_test


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
# 4. OpenStreetMap route optimization
# -----------------------------------------------------------
def build_osm_graph(city_name: str):
    """
    Download the drivable road network for a city from OpenStreetMap.
    """
    print(f"\n[OSM] Downloading road network for {city_name} from OpenStreetMap...")
    G = ox.graph_from_place(city_name, network_type="drive")
    # Add speed (km/h) and travel_time (seconds) estimates to edges
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)
    return G


def coords_to_nodes(G, coords_df):
    """
    Map each delivery's latitude/longitude to the nearest OSM street node.
    Requires 'Delivery_location_latitude' and 'Delivery_location_longitude'.
    """
    required_cols = [
        "Delivery_location_latitude",
        "Delivery_location_longitude",
    ]
    for c in required_cols:
        if c not in coords_df.columns:
            raise KeyError(
                f"Column '{c}' not found in coords. "
                "Check your CSV and update column names."
            )

    lats = coords_df["Delivery_location_latitude"].values
    lons = coords_df["Delivery_location_longitude"].values

    # OSMnx helper: find nearest street nodes for each lat/lon pair
    nodes = ox.distance.nearest_nodes(G, X=lons, Y=lats)
    return nodes


def route_distance_and_time(G, node_order):
    """
    Compute total route distance (meters) and time (seconds)
    for a sequence of nodes, using OSM shortest paths.
    """
    total_len = 0.0
    total_time = 0.0

    for u, v in zip(node_order[:-1], node_order[1:]):
        path = nx.shortest_path(G, u, v, weight="travel_time")
        edge_attrs = ox.utils_graph.get_route_edge_attributes(
            G, path, ["length", "travel_time"]
        )
        total_len += sum(e["length"] for e in edge_attrs)
        total_time += sum(e["travel_time"] for e in edge_attrs)

    return total_len, total_time


def nearest_neighbor_route(G, depot_node, stop_nodes):
    """
    Very simple route optimizer:
    greedy nearest-neighbor heuristic using travel_time on the road network.

    This is NOT an exact VRP solver, but it's enough to illustrate
    routing optimization for your proof-of-concept.
    """
    unvisited = list(stop_nodes)
    route = [depot_node]
    current = depot_node

    while unvisited:
        # choose next stop as the one with shortest path time from current
        next_idx = min(
            range(len(unvisited)),
            key=lambda i: nx.shortest_path_length(
                G, current, unvisited[i], weight="travel_time"
            ),
        )
        current = unvisited.pop(next_idx)
        route.append(current)

    # return to depot
    route.append(depot_node)

    return route


def compare_routes_with_osm(coords_sample, city_name: str):
    """
    Use OpenStreetMap to compare baseline vs optimized route distance/time:

      1) Build road network for the city.
      2) Map each delivery point to the nearest street node.
      3) Baseline route: visit deliveries in the original order.
      4) Optimized route: nearest-neighbor heuristic (shorter path).
    """
    if coords_sample is None:
        raise ValueError("No coordinates available for routing.")

    G = build_osm_graph(city_name)

    # Nodes for the delivery points
    stop_nodes = coords_to_nodes(G, coords_sample)

    # Define a 'depot' as the average restaurant location
    dep_lat_col = "Restaurant_latitude"
    dep_lon_col = "Restaurant_longitude"

    if dep_lat_col not in coords_sample.columns or dep_lon_col not in coords_sample.columns:
        raise KeyError(
            "Restaurant latitude/longitude columns not found in coords_sample. "
            "Update dep_lat_col/dep_lon_col to match your dataset."
        )

    depot_lat = coords_sample[dep_lat_col].mean()
    depot_lon = coords_sample[dep_lon_col].mean()
    depot_node = ox.distance.nearest_nodes(G, X=[depot_lon], Y=[depot_lat])[0]

    # Baseline: depot -> stops in original order -> depot
    baseline_route_nodes = [depot_node] + list(stop_nodes) + [depot_node]
    baseline_dist, baseline_time = route_distance_and_time(G, baseline_route_nodes)

    # Optimized: nearest-neighbor order
    optimized_route_nodes = nearest_neighbor_route(G, depot_node, list(stop_nodes))
    optimized_dist, optimized_time = route_distance_and_time(G, optimized_route_nodes)

    osm_results = {
        "baseline_km": baseline_dist / 1000,
        "optimized_km": optimized_dist / 1000,
        "baseline_minutes": baseline_time / 60,
        "optimized_minutes": optimized_time / 60,
    }

    print("\n[OSM] Route comparison (using OpenStreetMap):")
    print(f"  Baseline distance:  {osm_results['baseline_km']:.2f} km")
    print(f"  Optimized distance: {osm_results['optimized_km']:.2f} km")
    print(f"  Baseline time:      {osm_results['baseline_minutes']:.1f} minutes")
    print(f"  Optimized time:     {osm_results['optimized_minutes']:.1f} minutes")

    return osm_results


# -----------------------------------------------------------
# 5. Main script tying everything together
# -----------------------------------------------------------
def main():
    # 1. Train model on one area (your "city")
    area = "Urban "  # adjust to match Area value in your dataset
    model, X_test, y_test, coords_test = train_model(area=area)

    # 2. Simulate 50 deliveries and compute time/fuel improvement
    results = simulate_50_deliveries(model, X_test, y_test, coords_test, random_state=1)

    # 3. Print summary + 15% success check
    print("\n=== Simulation Results (50 Deliveries) ===")
    print(f"Baseline avg time:  {results['baseline_avg_time']:.2f} minutes")
    print(f"Optimized avg time: {results['optimized_avg_time']:.2f} minutes")
    print(f"Improvement:        {results['improvement_pct']:.1f}%")

    if results["improvement_pct"] >= 15:
        print("Success: target of 15% improvement achieved.")
    else:
        print("Target not reached yet. Consider tuning the model (features, model, hyperparameters).")

    # 4. Show dashboard
    plot_dashboard(results)

    # 5. Use OpenStreetMap to illustrate route optimization based on lat/lon
    if results["coords_sample"] is not None:
        # Use a city that matches your data (change this to a realistic city)
        city_name = "Bangalore, India"
        compare_routes_with_osm(results["coords_sample"], city_name=city_name)
    else:
        print("\n[OSM] Skipping routing: no latitude/longitude columns found.")


if __name__ == "__main__":
    main()
