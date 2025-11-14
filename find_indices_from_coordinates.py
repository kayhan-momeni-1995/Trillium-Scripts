#!/usr/bin/env python3
import numpy as np
import argparse
import llc

EARTH_RADIUS_M = 6371000.0  # spherical Earth radius (meters)


def parse_coord_string(coord_str):
    """
    Parse a coordinate string into (lat, lon) in decimal degrees.

    Accepted formats:
      - "0N 140W"
      - "10.5S 20.25E"
      - "0  -140"
      - "0, -140"
    """
    s = coord_str.strip()

    # Try simple "lat lon" decimal form first
    try:
        parts = [float(p) for p in s.replace(",", " ").split()]
        if len(parts) == 2:
            return parts[0], parts[1]
    except ValueError:
        pass

    tokens = s.replace(",", " ").split()
    if len(tokens) != 2:
        raise ValueError(f"Could not parse coordinate string: {coord_str!r}")

    lat = _parse_single_angle(tokens[0], is_lat=True)
    lon = _parse_single_angle(tokens[1], is_lat=False)
    return lat, lon


def _parse_single_angle(token, is_lat=True):
    """
    Parse a single latitude or longitude token with optional hemisphere, e.g. "10N", "140W".
    """
    t = token.strip().upper()
    if not t:
        raise ValueError(f"Empty coordinate token: {token!r}")

    hemi = t[-1]
    if hemi in ("N", "S", "E", "W"):
        val_str = t[:-1]
        if not val_str:
            raise ValueError(f"Missing numeric value in token: {token!r}")
        value = float(val_str)
        if hemi in ("S", "W"):
            value = -value
    else:
        # pure numeric
        value = float(t)

    # basic sanity checks (not strict, just to catch obvious nonsense)
    if is_lat and not (-90.0 <= value <= 90.0):
        raise ValueError(f"Latitude out of range: {value}")
    if not is_lat and not (-360.0 <= value <= 360.0):
        raise ValueError(f"Longitude looks suspicious: {value}")

    return value


def haversine_distance(lat0_deg, lon0_deg, lat_deg, lon_deg, radius=EARTH_RADIUS_M):
    """
    Great-circle distance (meters) between (lat0, lon0) and (lat, lon),
    where lat_deg and lon_deg can be arrays.
    """
    lat0 = np.radians(lat0_deg)
    lon0 = np.radians(lon0_deg)
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)

    dlat = lat - lat0
    dlon = lon - lon0

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat0) * np.cos(lat) * np.sin(dlon / 2.0) ** 2
    # numeric safety
    a = np.clip(a, 0.0, 1.0)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return radius * c


def find_closest_index(xc, yc, target_lat, target_lon):
    """
    Given XC, YC grids (both shape (nx, ny) in degrees), and a target lat/lon,
    return:
        (i0, j0)      = 0-based indices
        (i1, j1)      = 1-based indices (Fortran-style)
        distance_m    = distance in meters
        grid_lat, grid_lon = lat/lon of the closest grid point
    """
    # Compute distance to every grid point
    dist = haversine_distance(target_lat, target_lon, yc, xc)

    # Handle NaNs if present
    if np.isnan(dist).any():
        dist = np.where(np.isnan(dist), np.inf, dist)

    flat_idx = np.argmin(dist)
    i0, j0 = np.unravel_index(flat_idx, dist.shape)

    i1 = i0 + 1
    j1 = j0 + 1
    distance_m = float(dist[i0, j0])
    grid_lat = float(yc[i0, j0])
    grid_lon = float(xc[i0, j0])

    return (i0, j0), (i1, j1), distance_m, grid_lat, grid_lon


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Find the MITgcm grid indices (XC, YC) closest to a given coordinate.\n\n"
            "Examples:\n"
            "  find_closest.py xc.bin yc.bin 90 \"0N 140W\"\n"
            "  find_closest.py xc.bin yc.bin 90 \"0 -140\""
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--xc", help="Path to XC file (little-endian real*4)")
    parser.add_argument("--yc", help="Path to YC file (little-endian real*4)")
    parser.add_argument("--nx", type=int, help="Face size in llc grid (grid is nx by nx*13)")
    parser.add_argument(
        "--coord",
        help='Target coordinate, e.g. "0N 140W" or "0 -140"',
    )

    args = parser.parse_args()

    # Parse target coordinate
    target_lat, target_lon = parse_coord_string(args.coord)

    # Read grids
    xc = llc.readbin(args.xc, [args.nx, args.nx*13])
    yc = llc.readbin(args.yc, [args.nx, args.nx*13])

    print("XC shape:", xc.shape, "dtype:", xc.dtype)
    print("YC shape:", yc.shape, "dtype:", yc.dtype)
    print("XC min/max:", np.nanmin(xc), np.nanmax(xc))
    print("YC min/max:", np.nanmin(yc), np.nanmax(yc))
    print("Any NaNs in XC?", np.isnan(xc).any())
    print("Any NaNs in YC?", np.isnan(yc).any())

    # Find closest grid point
    (i0, j0), (i1, j1), dist_m, grid_lat, grid_lon = find_closest_index(
        xc, yc, target_lat, target_lon
    )

    print("Target coordinate:")
    print(f"  lat = {target_lat:.6f} deg, lon = {target_lon:.6f} deg")
    print()
    print("Closest grid point:")
    print(f"  0-based indices:        i = {i0}, j = {j0}")
    print(f"  grid lat, lon:          {grid_lat:.6f} deg, {grid_lon:.6f} deg")
    print(f"  distance:               {dist_m:.3f} m")


if __name__ == "__main__":
    main()

