"""
Manganese Nodule Concentration Prediction Pipeline

Predicts manganese nodule concentrations (kg/m²) from bathymetry data
using XGBoost with Leave-One-Out Cross-Validation.
"""

from pathlib import Path

import folium
import numpy as np
import polars as pl
import xarray as xr
from folium.plugins import HeatMap
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import generic_filter
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut
from xgboost import XGBRegressor

# Paths
PROJECT_ROOT = Path(__file__).parent
LABELS_DIR = PROJECT_ROOT / "labels"
BATHY_FILE = PROJECT_ROOT / "bathy" / "gebco_2025_n20.0_s0.0_w-160.0_e-115.0.nc"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def load_label_data() -> pl.DataFrame:
    """Load and merge nodule concentration data from both tab files."""
    # Load Fewkes data
    fewkes = pl.read_csv(
        LABELS_DIR / "Fewkes_1980_2.tab",
        separator="\t",
    ).select([
        pl.col("Event"),
        pl.col("Latitude"),
        pl.col("Longitude"),
        pl.col("Elevation [m]").alias("elevation_m"),
        pl.col("Nodules [kg/m**2]").alias("concentration_kg_m2"),
    ])

    # Load Piper data
    piper = pl.read_csv(
        LABELS_DIR / "Piper-etal_1979_tab8.tab",
        separator="\t",
    ).select([
        pl.col("Event"),
        pl.col("Latitude"),
        pl.col("Longitude"),
        pl.col("Elevation [m]").alias("elevation_m"),
        pl.col("Conc [kg/m**2]").alias("concentration_kg_m2"),
    ])

    # Combine datasets
    combined = pl.concat([fewkes, piper])

    # Round coordinates for deduplication (3 decimal places ~ 100m precision)
    combined = combined.with_columns([
        pl.col("Latitude").round(3).alias("lat_rounded"),
        pl.col("Longitude").round(3).alias("lon_rounded"),
    ])

    # Deduplicate by averaging concentrations at same location
    deduplicated = combined.group_by(["lat_rounded", "lon_rounded"]).agg([
        pl.col("Event").first(),
        pl.col("Latitude").mean(),
        pl.col("Longitude").mean(),
        pl.col("elevation_m").mean(),
        pl.col("concentration_kg_m2").mean(),
    ]).drop(["lat_rounded", "lon_rounded"])

    # Filter out zero/null values
    deduplicated = deduplicated.filter(
        (pl.col("concentration_kg_m2") > 0) & pl.col("concentration_kg_m2").is_not_null()
    )

    print(f"Loaded {len(deduplicated)} unique sample locations")
    print(f"  Concentration range: {deduplicated['concentration_kg_m2'].min():.2f} - "
          f"{deduplicated['concentration_kg_m2'].max():.2f} kg/m²")
    print(f"  Lat range: {deduplicated['Latitude'].min():.2f}° - {deduplicated['Latitude'].max():.2f}°N")
    print(f"  Lon range: {deduplicated['Longitude'].min():.2f}° - {deduplicated['Longitude'].max():.2f}°W")

    return deduplicated


def load_bathymetry():
    """Load GEBCO bathymetry data and create interpolators."""
    print("\nLoading bathymetry data...")
    ds = xr.open_dataset(BATHY_FILE)

    # GEBCO uses 'elevation' variable (negative for depth)
    elevation = ds["elevation"].values
    lat = ds["lat"].values
    lon = ds["lon"].values

    print(f"  Grid size: {elevation.shape}")
    print(f"  Lat range: {lat.min():.2f}° - {lat.max():.2f}°")
    print(f"  Lon range: {lon.min():.2f}° - {lon.max():.2f}°")

    # Create depth interpolator (convert elevation to positive depth)
    depth_interp = RegularGridInterpolator(
        (lat, lon), -elevation,  # Negate to get positive depth
        method="linear",
        bounds_error=False,
        fill_value=np.nan
    )

    # Compute slope (gradient magnitude)
    # GEBCO resolution is ~15 arc-seconds = ~450m at equator
    dlat = np.abs(np.mean(np.diff(lat)))
    dlon = np.abs(np.mean(np.diff(lon)))
    meters_per_degree_lat = 111320  # approximate
    meters_per_degree_lon = 111320 * np.cos(np.radians(10))  # at ~10°N

    grad_lat = np.gradient(elevation, dlat * meters_per_degree_lat, axis=0)
    grad_lon = np.gradient(elevation, dlon * meters_per_degree_lon, axis=1)
    slope = np.sqrt(grad_lat**2 + grad_lon**2)

    slope_interp = RegularGridInterpolator(
        (lat, lon), slope,
        method="linear",
        bounds_error=False,
        fill_value=np.nan
    )

    # Compute roughness (local std deviation in 5x5 window)
    def local_std(arr):
        return np.std(arr)

    print("  Computing roughness (this may take a moment)...")
    roughness = generic_filter(elevation, local_std, size=5, mode='nearest')

    roughness_interp = RegularGridInterpolator(
        (lat, lon), roughness,
        method="linear",
        bounds_error=False,
        fill_value=np.nan
    )

    # Compute aspect (terrain orientation)
    aspect = np.arctan2(grad_lon, grad_lat)  # radians
    aspect_sin = np.sin(aspect)
    aspect_cos = np.cos(aspect)

    aspect_sin_interp = RegularGridInterpolator(
        (lat, lon), aspect_sin,
        method="linear",
        bounds_error=False,
        fill_value=np.nan
    )

    aspect_cos_interp = RegularGridInterpolator(
        (lat, lon), aspect_cos,
        method="linear",
        bounds_error=False,
        fill_value=np.nan
    )

    ds.close()

    return {
        "depth": depth_interp,
        "slope": slope_interp,
        "roughness": roughness_interp,
        "aspect_sin": aspect_sin_interp,
        "aspect_cos": aspect_cos_interp,
        "lat": lat,
        "lon": lon,
        "elevation": elevation,
    }


def extract_features(df: pl.DataFrame, bathy: dict) -> np.ndarray:
    """Extract bathymetry features for sample locations."""
    lats = df["Latitude"].to_numpy()
    lons = df["Longitude"].to_numpy()
    points = np.column_stack([lats, lons])

    # Primary features from bathymetry
    depth = bathy["depth"](points)
    slope = bathy["slope"](points)
    roughness = bathy["roughness"](points)
    aspect_sin = bathy["aspect_sin"](points)
    aspect_cos = bathy["aspect_cos"](points)

    # Derived features
    depth_squared = depth ** 2
    depth_log = np.log(depth + 1)  # +1 to avoid log(0)

    # Latitude/longitude encoding (cyclical)
    lat_sin = np.sin(np.radians(lats))
    lat_cos = np.cos(np.radians(lats))
    lon_sin = np.sin(np.radians(lons))
    lon_cos = np.cos(np.radians(lons))

    # Depth zone indicator (optimal range 4000-6000m)
    in_optimal_zone = ((depth >= 4000) & (depth <= 6000)).astype(float)

    features = np.column_stack([
        depth,
        slope,
        roughness,
        aspect_sin,
        aspect_cos,
        depth_squared,
        depth_log,
        lat_sin,
        lat_cos,
        lon_sin,
        lon_cos,
        in_optimal_zone,
    ])

    return features


FEATURE_NAMES = [
    "depth_m",
    "slope",
    "roughness_m",
    "aspect_sin",
    "aspect_cos",
    "depth_squared",
    "depth_log",
    "lat_sin",
    "lat_cos",
    "lon_sin",
    "lon_cos",
    "in_optimal_zone",
]


def create_sample_map(df: pl.DataFrame) -> None:
    """Create interactive map of sample locations."""
    print("\nCreating sample location map...")

    # Center on Pacific nodule field
    center_lat = df["Latitude"].mean()
    center_lon = df["Longitude"].mean()

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=5,
        tiles="CartoDB positron",
    )

    # Color scale for concentrations
    conc_min = df["concentration_kg_m2"].min()
    conc_max = df["concentration_kg_m2"].max()

    def get_color(conc):
        """Map concentration to color (green=low, yellow=medium, red=high)."""
        normalized = (conc - conc_min) / (conc_max - conc_min)
        if normalized < 0.33:
            return "green"
        elif normalized < 0.66:
            return "orange"
        else:
            return "red"

    # Add markers for each sample
    for row in df.iter_rows(named=True):
        color = get_color(row["concentration_kg_m2"])
        popup_text = (
            f"<b>Event:</b> {row['Event']}<br>"
            f"<b>Concentration:</b> {row['concentration_kg_m2']:.2f} kg/m²<br>"
            f"<b>Depth:</b> {abs(row['elevation_m']):.0f} m<br>"
            f"<b>Location:</b> {row['Latitude']:.3f}°N, {row['Longitude']:.3f}°W"
        )

        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=6,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7,
            popup=folium.Popup(popup_text, max_width=300),
        ).add_to(m)

    # Add legend
    legend_html = """
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000;
                background-color: white; padding: 10px; border: 2px solid gray;
                border-radius: 5px; font-family: Arial;">
        <b>Nodule Concentration (kg/m²)</b><br>
        <i style="background: green; width: 12px; height: 12px;
           display: inline-block; border-radius: 50%;"></i> Low (0-8)<br>
        <i style="background: orange; width: 12px; height: 12px;
           display: inline-block; border-radius: 50%;"></i> Medium (8-16)<br>
        <i style="background: red; width: 12px; height: 12px;
           display: inline-block; border-radius: 50%;"></i> High (16-26)
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    m.save(OUTPUT_DIR / "sample_locations.html")
    print(f"  Saved to {OUTPUT_DIR / 'sample_locations.html'}")


def train_model(X: np.ndarray, y: np.ndarray) -> tuple:
    """Train XGBoost with Leave-One-Out CV."""
    print("\nTraining XGBoost model with Leave-One-Out CV...")

    # Remove samples with NaN features
    valid_mask = ~np.any(np.isnan(X), axis=1)
    X_valid = X[valid_mask]
    y_valid = y[valid_mask]
    print(f"  Using {len(y_valid)} samples (removed {len(y) - len(y_valid)} with missing features)")

    # Model with regularization for small dataset
    model = XGBRegressor(
        n_estimators=100,
        max_depth=4,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )

    # Leave-One-Out CV
    loo = LeaveOneOut()
    predictions = np.zeros(len(y_valid))

    for train_idx, test_idx in loo.split(X_valid):
        X_train, X_test = X_valid[train_idx], X_valid[test_idx]
        y_train = y_valid[train_idx]

        model.fit(X_train, y_train)
        predictions[test_idx] = model.predict(X_test)

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_valid, predictions))
    mae = mean_absolute_error(y_valid, predictions)
    r2 = r2_score(y_valid, predictions)

    print("\n  Leave-One-Out CV Results:")
    print(f"    RMSE: {rmse:.3f} kg/m²")
    print(f"    MAE:  {mae:.3f} kg/m²")
    print(f"    R²:   {r2:.3f}")

    # Train final model on all data
    model.fit(X_valid, y_valid)

    # Feature importance
    importance = dict(zip(FEATURE_NAMES, model.feature_importances_))
    importance_sorted = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    print("\n  Feature Importance:")
    for name, imp in importance_sorted:
        print(f"    {name}: {imp:.3f}")

    # Save metrics to file
    with open(OUTPUT_DIR / "model_metrics.txt", "w") as f:
        f.write("Manganese Nodule Concentration Prediction Model\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Training samples: {len(y_valid)}\n")
        f.write(f"Features: {len(FEATURE_NAMES)}\n\n")
        f.write("Leave-One-Out Cross-Validation Results:\n")
        f.write(f"  RMSE: {rmse:.3f} kg/m²\n")
        f.write(f"  MAE:  {mae:.3f} kg/m²\n")
        f.write(f"  R²:   {r2:.3f}\n\n")
        f.write("Feature Importance:\n")
        for name, imp in importance_sorted:
            f.write(f"  {name}: {imp:.3f}\n")

    print(f"  Saved metrics to {OUTPUT_DIR / 'model_metrics.txt'}")

    return model, valid_mask, {"rmse": rmse, "mae": mae, "r2": r2}


def create_prediction_map(model, bathy: dict, df: pl.DataFrame, valid_mask: np.ndarray) -> None:
    """Generate prediction heatmap over study area."""
    print("\nGenerating prediction map...")

    # Create grid at 0.5° resolution over study area
    lat_range = np.arange(5, 18, 0.5)
    lon_range = np.arange(-155, -120, 0.5)

    grid_lats, grid_lons = np.meshgrid(lat_range, lon_range, indexing="ij")
    grid_points = np.column_stack([grid_lats.ravel(), grid_lons.ravel()])

    # Extract features at grid points
    depth = bathy["depth"](grid_points)
    slope = bathy["slope"](grid_points)
    roughness = bathy["roughness"](grid_points)
    aspect_sin = bathy["aspect_sin"](grid_points)
    aspect_cos = bathy["aspect_cos"](grid_points)

    depth_squared = depth ** 2
    depth_log = np.log(np.maximum(depth, 1))

    lat_sin = np.sin(np.radians(grid_points[:, 0]))
    lat_cos = np.cos(np.radians(grid_points[:, 0]))
    lon_sin = np.sin(np.radians(grid_points[:, 1]))
    lon_cos = np.cos(np.radians(grid_points[:, 1]))

    in_optimal_zone = ((depth >= 4000) & (depth <= 6000)).astype(float)

    X_grid = np.column_stack([
        depth, slope, roughness, aspect_sin, aspect_cos,
        depth_squared, depth_log, lat_sin, lat_cos, lon_sin, lon_cos,
        in_optimal_zone
    ])

    # Predict
    predictions = model.predict(X_grid)
    predictions = np.clip(predictions, 0, None)  # No negative concentrations

    # Filter to deep ocean (>3000m) and valid data
    ocean_mask = (depth > 3000) & ~np.isnan(depth)

    # Prepare heatmap data
    heat_data = []
    for i in range(len(grid_points)):
        if ocean_mask[i] and predictions[i] > 0.5:
            heat_data.append([
                grid_points[i, 0],
                grid_points[i, 1],
                float(predictions[i])
            ])

    print(f"  Created {len(heat_data)} prediction points")

    # Create map
    center_lat = df["Latitude"].mean()
    center_lon = df["Longitude"].mean()

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=5,
        tiles="CartoDB positron",
    )

    # Add heatmap layer
    if heat_data:
        HeatMap(
            heat_data,
            min_opacity=0.3,
            radius=15,
            blur=10,
            gradient={
                0.2: "blue",
                0.4: "cyan",
                0.6: "lime",
                0.8: "yellow",
                1.0: "red"
            }
        ).add_to(m)

    # Overlay actual sample points
    df_valid = df.filter(pl.Series(valid_mask))

    for row in df_valid.iter_rows(named=True):
        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=5,
            color="black",
            fill=True,
            fillColor="white",
            fillOpacity=0.8,
            popup=f"{row['Event']}: {row['concentration_kg_m2']:.1f} kg/m²",
        ).add_to(m)

    # Add legend
    legend_html = """
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000;
                background-color: white; padding: 10px; border: 2px solid gray;
                border-radius: 5px; font-family: Arial;">
        <b>Predicted Concentration</b><br>
        <div style="background: linear-gradient(to right, blue, cyan, lime, yellow, red);
                    width: 100px; height: 15px; margin: 5px 0;"></div>
        <span>Low → High (kg/m²)</span><br><br>
        <i style="background: white; border: 2px solid black; width: 10px; height: 10px;
           display: inline-block; border-radius: 50%;"></i> Actual samples
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    m.save(OUTPUT_DIR / "predictions.html")
    print(f"  Saved to {OUTPUT_DIR / 'predictions.html'}")


def main():
    """Run the manganese nodule prediction pipeline."""
    print("=" * 60)
    print("Manganese Nodule Concentration Prediction Pipeline")
    print("=" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Step 1: Load label data
    df = load_label_data()

    # Step 2: Load bathymetry and create interpolators
    bathy = load_bathymetry()

    # Step 3: Extract features
    print("\nExtracting bathymetry features...")
    X = extract_features(df, bathy)
    y = df["concentration_kg_m2"].to_numpy()
    print(f"  Feature matrix shape: {X.shape}")

    # Step 4: Create sample location map
    create_sample_map(df)

    # Step 5: Train model
    model, valid_mask, metrics = train_model(X, y)

    # Step 6: Generate prediction map
    create_prediction_map(model, bathy, df, valid_mask)

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print(f"  - Sample map: {OUTPUT_DIR / 'sample_locations.html'}")
    print(f"  - Prediction map: {OUTPUT_DIR / 'predictions.html'}")
    print(f"  - Model metrics: {OUTPUT_DIR / 'model_metrics.txt'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
