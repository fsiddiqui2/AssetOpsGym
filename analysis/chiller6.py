"""Is AssetOpsBench's Chiller 6 telemetry real? (T5)

    uv run --no-project --with numpy --with scipy \
      --with matplotlib python analysis/chiller6.py

Prints the stats in chiller6.md and writes chiller6.png.
"""

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

SRC = (Path(__file__).parents[2] / "AssetOpsBench/src/couchdb"
       / "scenarios_data/shared/iot/chiller_6.json")
OUT = Path(__file__).with_suffix(".png")
STEP = 900  # s, the regular grid
SENSORS = ["Tonnage", "Power Input", "Chiller Efficiency",
           "Chiller % Loaded", "Supply Temperature",
           "Return Temperature", "Setpoint Temperature",
           "Condenser Water Flow",
           "Condenser Water Return To Tower Temperature",
           "Liquid Refrigerant Evaporator Temperature"]


def load() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Regular 15-min rows with every sensor, on a full grid.

    Missing slots are NaN. The ~20 irregular event rows
    appended to the file (Run Status, partial readings)
    are dropped.
    """
    raw = json.load(open(SRC))
    keys = ["Chiller 6 " + s for s in SENSORS]
    rows = [r for r in raw if all(k in r for k in keys)
            and r["timestamp"][-5:] in ("00:00", "15:00",
                                        "30:00", "45:00")]
    t = np.array([datetime.fromisoformat(r["timestamp"])
                  .timestamp() for r in rows])
    i = ((t - t.min()) // STEP).astype(int)
    grid = t.min() + STEP * np.arange(i.max() + 1)
    x = {}
    for s, k in zip(SENSORS, keys):
        x[s] = np.full(len(grid), np.nan)
        x[s][i] = [r[k] for r in rows]
    return grid, x


def acf(x: np.ndarray, lag: int) -> float:
    """Autocorrelation over pairs where both ends are present."""
    m = np.nanmean(x)
    a, b = x[:-lag] - m, x[lag:] - m
    ok = ~np.isnan(a) & ~np.isnan(b)
    return float(np.mean(a[ok] * b[ok]) / np.nanvar(x))


def main() -> None:
    grid, x = load()
    on = x["Tonnage"] > 0
    run = {s: np.where(on, v, np.nan) for s, v in x.items()}
    n_pairs = int(np.sum(on[:-1] & on[1:]))
    band = 2 / np.sqrt(n_pairs)
    present = ~np.isnan(x["Tonnage"])
    print(f"{present.sum()} rows, {on.sum()} running, "
          f"{n_pairs} running pairs, white band ±{band:.3f}")

    print(f"{'sensor':44s} {'median':>8s} {'p99':>8s} "
          f"{'acf1':>6s} {'acf1 on':>7s}")
    for s in SENSORS:
        v = run[s][on]
        print(f"{s:44s} {np.median(v):8.2f} "
              f"{np.percentile(v, 99):8.2f} "
              f"{acf(x[s], 1):6.3f} {acf(run[s], 1):7.3f}")

    sup, ret = x["Supply Temperature"], x["Return Temperature"]
    p, e, t = x["Power Input"], x["Chiller Efficiency"], x["Tonnage"]
    print(f"return < supply while running: "
          f"{np.mean(ret[on] < sup[on]):.1%}")
    print(f"setpoint std while running: "
          f"{np.std(x['Setpoint Temperature'][on]):.1f} F")
    print(f"corr(power, kW/ton x tons): "
          f"{np.corrcoef(p[on], (e * t)[on])[0, 1]:.3f}")
    print(f"corr(tons, % loaded): "
          f"{np.corrcoef(t[on], x['Chiller % Loaded'][on])[0, 1]:.3f}")
    hour = np.array([datetime.fromtimestamp(g).hour for g in grid])
    groups = [t[on & (hour == h)] for h in range(24)]
    print(f"daily load cycle, ANOVA p: "
          f"{stats.f_oneway(*groups).pvalue:.2f}")

    d = np.diff(np.r_[0, on.astype(int), 0])
    starts = np.flatnonzero(d == 1)
    lens = (np.flatnonzero(d == -1) - starts) * STEP / 3600
    start_h = [datetime.fromtimestamp(grid[i]).hour for i in starts]
    print(f"run start hours: {sorted(start_h)}")
    print(f"on runs: {len(starts)}, median {np.median(lens):.1f} h, "
          f"max {lens.max():.1f} h")

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    days = slice(0, 3 * 96)
    hrs = (grid[days] - grid[0]) / 3600
    a = ax[0, 0]
    a.plot(hrs, t[days], lw=.8, label="Tonnage")
    a.plot(hrs, 20 * ret[days], lw=.8, label="Return temp ×20")
    a.set(title="First 3 days", xlabel="hour", ylabel="value")
    a.legend()
    a = ax[0, 1]
    lags = np.arange(1, 193)
    for s in ["Tonnage", "Power Input", "Return Temperature",
              "Setpoint Temperature"]:
        a.plot(lags / 4, [acf(run[s], k) for k in lags], lw=.8,
               label=s)
    a.axhspan(-band, band, color="grey", alpha=.3,
              label="white noise ±2σ")
    a.set(title="Autocorrelation while running",
          xlabel="lag (h)", ylim=(-.3, 1))
    a.legend(fontsize=8)
    a = ax[1, 0]
    a.scatter((e * t)[on], p[on], s=3, alpha=.4)
    lim = np.nanmax(p[on])
    a.plot([0, lim], [0, lim], "k--", lw=1, label="power = kW/ton × tons")
    a.set(title="Power vs efficiency × tonnage",
          xlabel="kW/ton × tons", ylabel="Power Input (kW)")
    a.legend()
    a = ax[1, 1]
    a.scatter(sup[on], ret[on], s=3, alpha=.4)
    lim = np.nanmax(ret[on])
    a.plot([0, lim], [0, lim], "k--", lw=1)
    a.set(title="Chilled water: below the line is impossible",
          xlabel="Supply temp (F)", ylabel="Return temp (F)")
    fig.tight_layout()
    fig.savefig(OUT, dpi=130)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
