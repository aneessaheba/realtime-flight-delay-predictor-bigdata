"""
generate_sample_data.py
-----------------------
Creates a synthetic BTS-schema CSV for local testing when you don't
have the full dataset downloaded yet.  Produces ~100k rows spanning
2018-2023 with realistic class imbalance (~20% delayed).

Run:
    python src/generate_sample_data.py
"""

import os
import numpy as np
import pandas as pd

SEED = 42
N_ROWS = 100_000
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sample_flights.csv")

CARRIERS = ["AA", "DL", "UA", "WN", "AS", "B6", "NK", "F9", "G4", "HA"]
AIRPORTS = [
    "ATL", "LAX", "ORD", "DFW", "DEN", "JFK", "SFO", "SEA", "LAS", "MCO",
    "EWR", "CLT", "PHX", "IAH", "MIA", "BOS", "MSP", "DTW", "FLL", "PHL",
]


def main():
    rng = np.random.default_rng(SEED)

    years = rng.integers(2018, 2024, size=N_ROWS)
    months = rng.integers(1, 13, size=N_ROWS)
    day_of_month = rng.integers(1, 29, size=N_ROWS)
    day_of_week = rng.integers(1, 8, size=N_ROWS)

    carriers = rng.choice(CARRIERS, size=N_ROWS)
    origins = rng.choice(AIRPORTS, size=N_ROWS)
    dests = rng.choice(AIRPORTS, size=N_ROWS)

    crs_dep_time = rng.integers(500, 2359, size=N_ROWS)
    crs_arr_time = rng.integers(600, 2359, size=N_ROWS)
    crs_elapsed = rng.integers(60, 480, size=N_ROWS).astype(float)
    distance = rng.integers(100, 3000, size=N_ROWS).astype(float)

    dep_delay = rng.normal(5, 20, size=N_ROWS)
    dep_delay = np.clip(dep_delay, -30, 300)

    arr_delay = dep_delay + rng.normal(0, 10, size=N_ROWS)
    arr_delay = np.clip(arr_delay, -60, 360)

    # Delay cause columns (null for on-time flights)
    is_delayed = arr_delay > 15
    carrier_delay = np.where(is_delayed, rng.exponential(15, N_ROWS), 0.0)
    weather_delay = np.where(is_delayed, rng.exponential(5, N_ROWS), 0.0)
    nas_delay = np.where(is_delayed, rng.exponential(8, N_ROWS), 0.0)
    security_delay = np.where(is_delayed, rng.exponential(1, N_ROWS), 0.0)
    late_aircraft = np.where(is_delayed, rng.exponential(10, N_ROWS), 0.0)

    df = pd.DataFrame({
        "YEAR": years,
        "MONTH": months,
        "DAY_OF_MONTH": day_of_month,
        "DAY_OF_WEEK": day_of_week,
        "OP_UNIQUE_CARRIER": carriers,
        "ORIGIN": origins,
        "DEST": dests,
        "CRS_DEP_TIME": crs_dep_time,
        "DEP_DELAY": dep_delay.round(1),
        "CRS_ARR_TIME": crs_arr_time,
        "ARR_DELAY": arr_delay.round(1),
        "CRS_ELAPSED_TIME": crs_elapsed,
        "DISTANCE": distance,
        "CARRIER_DELAY": carrier_delay.round(1),
        "WEATHER_DELAY": weather_delay.round(1),
        "NAS_DELAY": nas_delay.round(1),
        "SECURITY_DELAY": security_delay.round(1),
        "LATE_AIRCRAFT_DELAY": late_aircraft.round(1),
    })

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    delayed_pct = is_delayed.mean() * 100
    print(f"Generated {N_ROWS:,} rows → {OUT_PATH}")
    print(f"Class balance: {delayed_pct:.1f}% delayed, {100-delayed_pct:.1f}% on-time")


if __name__ == "__main__":
    main()
