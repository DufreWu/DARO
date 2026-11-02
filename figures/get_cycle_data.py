"""
get_cycle_data.py
=================
Extract Rainflow cycle data from battery SOC time-series and compute
per-cycle degradation stress factors (Eqs. 8a–8c, DARO paper).

Three usage modes
-----------------
A. From a CSV/numpy file containing a measured SOC trace.
B. From a simulated multi-mission SOC trace (useful for sweeping speeds).
C. Quick demo that reproduces the Fig 3(b) degradation analysis.

Requirements
------------
    pip install rainflow numpy pandas matplotlib scipy
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rainflow                          # pip install rainflow
from pathlib import Path
from typing import Sequence


# ══════════════════════════════════════════════════════════════════════════════
#  Battery degradation model parameters  (Table I, DARO paper)
# ══════════════════════════════════════════════════════════════════════════════
K_SIGMA   = 1.04
K_T       = 6.93e-2
K_D1      = 1.40e5
K_D2      = -5.01e-1
K_D3      = -1.23e5
SIGMA_REF = 0.50    # reference average SOC
T_REF_K   = 298.15  # reference temperature, K  (25 °C)

MODEL_VARIANTS = {
    "full": {"avg_soc": True, "dod": True, "temperature": True},
    "without_dod": {"avg_soc": True, "dod": False, "temperature": True},
    "without_avg_soc": {"avg_soc": False, "dod": True, "temperature": True},
    "without_temperature": {"avg_soc": True, "dod": True, "temperature": False},
}


# ══════════════════════════════════════════════════════════════════════════════
#  Stress factor functions  (Eqs. 8a – 8c)
# ══════════════════════════════════════════════════════════════════════════════
def S_sigma(avg_soc: float | np.ndarray) -> float | np.ndarray:
    """Average-SOC stress factor (Eq. 8a)."""
    return np.exp(K_SIGMA * (avg_soc - SIGMA_REF))


def S_delta(dod: float | np.ndarray) -> float | np.ndarray:
    """Depth-of-Discharge stress factor (Eq. 8b).
    dod in [0, 1]; returns very small positive floats (~1e-5 range).
    """
    return (K_D1 * np.asarray(dod) ** K_D2 + K_D3) ** (-1)


def S_T(T_celsius: float | np.ndarray) -> float | np.ndarray:
    """Temperature stress factor (Eq. 8c)."""
    T_K = np.asarray(T_celsius) + 273.15
    return np.exp(K_T * (T_K - T_REF_K) * T_REF_K / T_K)


def stress_factors(
        dod: float | np.ndarray,
        avg_soc: float | np.ndarray,
        T_celsius: float | np.ndarray,
        model: str = "full",
) -> tuple[float | np.ndarray, float | np.ndarray, float | np.ndarray]:
    """Return the three stress factors, optionally disabling selected terms."""
    if model not in MODEL_VARIANTS:
        raise ValueError(f"Unsupported model '{model}'. Choose from {sorted(MODEL_VARIANTS)}")

    cfg = MODEL_VARIANTS[model]
    sigma = float(S_sigma(avg_soc)) if np.isscalar(avg_soc) else S_sigma(avg_soc)
    delta = float(S_delta(dod)) if np.isscalar(dod) else S_delta(dod)
    temp = float(S_T(T_celsius)) if np.isscalar(T_celsius) else S_T(T_celsius)

    if not cfg["avg_soc"]:
        sigma = 1.0 if np.isscalar(sigma) else np.ones_like(sigma)
    if not cfg["dod"]:
        delta = 1.0 if np.isscalar(delta) else np.ones_like(delta)
    if not cfg["temperature"]:
        temp = 1.0 if np.isscalar(temp) else np.ones_like(temp)
    return sigma, delta, temp


def degradation_per_cycle(
        dod: float,
        avg_soc: float,
        T_celsius: float = 25.0,
        is_full: bool = True,
        model: str = "full") -> float:
    """
    Per-cycle degradation contribution  f_d^(i) = n_i · S_σ · S_δ · S_T
    (Eq. 7, DARO paper).

    Parameters
    ----------
    dod        : Depth of Discharge for this cycle  (fraction, 0–1).
    avg_soc    : Average SOC during the cycle        (fraction, 0–1).
    T_celsius  : Average battery temperature         (°C).
    is_full    : True  → n_i = 1.0 (full cycle)
                 False → n_i = 0.5 (half cycle, as flagged by Rainflow)
    model      : Degradation model variant: "full", "without_dod",
                 "without_avg_soc", or "without_temperature".
    """
    n_i = 1.0 if is_full else 0.5
    sigma, delta, temp = stress_factors(dod, avg_soc, T_celsius, model=model)
    return n_i * sigma * delta * temp


# ══════════════════════════════════════════════════════════════════════════════
#  Rainflow cycle extraction
# ══════════════════════════════════════════════════════════════════════════════
def extract_rainflow_cycles(
        soc_trace: np.ndarray,
        T_trace:   np.ndarray | None = None,
        min_dod:   float = 0.01,
        model:     str = "full",
) -> pd.DataFrame:
    """
    Run Rainflow counting on a SOC time-series and return a DataFrame
    of all detected cycles with their stress factors.

    Parameters
    ----------
    soc_trace : 1-D array of SOC values in [0, 1], sampled at any
                fixed or variable rate (only the turning-point sequence
                matters for Rainflow).
    T_trace   : Temperature array matching soc_trace.  If None,
                assumes constant 25 °C throughout.
    min_dod   : Ignore micro-cycles below this DOD threshold
                (eliminates sensor noise, default 1%).
    model     : Degradation model variant: "full", "without_dod",
                "without_avg_soc", or "without_temperature".

    Returns
    -------
    DataFrame with columns:
        dod        – cycle depth of discharge
        avg_soc    – mean SOC over the cycle
        count      – 0.5 (half) or 1.0 (full) cycle
        T_avg      – average temperature during the cycle (°C)
        S_sigma    – SOC stress factor
        S_delta    – DOD stress factor
        S_T        – temperature stress factor
        fd_cycle   – combined degradation contribution  (= count·Sσ·Sδ·ST)
        i_start    – index in soc_trace where cycle starts
        i_end      – index in soc_trace where cycle ends
    """
    soc_trace = np.asarray(soc_trace, dtype=float)
    if T_trace is None:
        T_trace = np.full_like(soc_trace, 25.0)
    else:
        T_trace = np.asarray(T_trace, dtype=float)

    rows = []
    for rng, mean, count, i_start, i_end in rainflow.extract_cycles(soc_trace):
        dod = abs(rng)           # range = peak-to-valley SOC swing = DOD
        if dod < min_dod:
            continue

        # average temperature over the cycle span
        i0, i1 = int(i_start), int(i_end)
        T_avg = float(np.mean(T_trace[min(i0, i1): max(i0, i1) + 1]))

        Ss, Sd, ST = stress_factors(dod, mean, T_avg, model=model)
        fd = float(count * Ss * Sd * ST)

        rows.append({
            "dod":     dod,
            "avg_soc": mean,
            "count":   count,
            "T_avg":   T_avg,
            "S_sigma": float(Ss),
            "S_delta": float(Sd),
            "S_T":     float(ST),
            "fd_cycle": fd,
            "i_start": i_start,
            "i_end":   i_end,
            "model":   model,
        })

    return pd.DataFrame(rows)


def summarise_cycles(df: pd.DataFrame) -> dict:
    """
    Aggregate cycle DataFrame into scalar health metrics.

    Returns
    -------
    dict with:
        n_cycles       – total equivalent full cycles
        fd_total       – cumulative degradation (sum of fd_cycle)
        soh_remaining  – estimated SOH = exp(-fd_total)
        missions_to_eol– predicted missions to SOH=0.80  (if called per-mission)
        dod_mean       – weighted-average DOD
        dod_std        – standard deviation of DOD
    """
    if df.empty:
        return {}

    n_cycles  = df["count"].sum()
    fd_total  = df["fd_cycle"].sum()
    soh       = float(np.exp(-fd_total))
    # missions to EOL: -ln(0.80) / fd_per_mission
    fd_eol    = -np.log(0.80)

    dod_mean  = np.average(df["dod"], weights=df["count"])
    dod_std   = np.sqrt(np.average((df["dod"] - dod_mean) ** 2,
                                    weights=df["count"]))
    return {
        "n_cycles":        n_cycles,
        "fd_total":        fd_total,
        "soh_remaining":   soh,
        "fd_to_eol":       fd_eol,
        "dod_mean":        dod_mean,
        "dod_std":         dod_std,
    }


def compare_model_variants(
        soc_trace: np.ndarray,
        T_trace: np.ndarray | None = None,
        min_dod: float = 0.01,
) -> pd.DataFrame:
    """Return a summary table across the four degradation-model variants."""
    rows = []
    for model in ["full", "without_dod", "without_avg_soc", "without_temperature"]:
        df = extract_rainflow_cycles(soc_trace, T_trace, min_dod=min_dod, model=model)
        stats = summarise_cycles(df)
        rows.append({
            "model": model,
            "n_cycles": stats.get("n_cycles", 0),
            "fd_total": stats.get("fd_total", 0.0),
            "soh_remaining": stats.get("soh_remaining", 1.0),
            "dod_mean": stats.get("dod_mean", np.nan),
            "dod_std": stats.get("dod_std", np.nan),
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
#  Mode A – load from file
# ══════════════════════════════════════════════════════════════════════════════
def cycles_from_file(
        path: str | Path,
        soc_col:  str = "soc",
        time_col: str | None = "time",
        T_col:    str | None = None,
        model:    str = "full",
        **read_kw,
) -> pd.DataFrame:
    """
    Load a SOC trace from CSV (or any pandas-readable format) and
    return the Rainflow cycle DataFrame.

    CSV must contain at least a SOC column in [0, 1].
    Optionally a temperature column (°C) for per-cycle S_T computation.

    Example CSV layout
    ------------------
    time,soc,temperature
    0.0,1.000,24.5
    1.0,0.998,24.6
    ...

    Usage
    -----
    df_cycles = cycles_from_file("soc_log.csv", soc_col="soc", T_col="temperature")
    print(summarise_cycles(df_cycles))
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        raw = pd.read_csv(path, **read_kw)
    elif suffix in (".xlsx", ".xls"):
        raw = pd.read_excel(path, **read_kw)
    elif suffix == ".npy":
        arr = np.load(path)
        # assume shape (N,) → SOC only, or (N,2) → [time, SOC]
        if arr.ndim == 1:
            raw = pd.DataFrame({"soc": arr})
        else:
            raw = pd.DataFrame(arr, columns=["time", "soc"])
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    soc_trace = raw[soc_col].to_numpy(dtype=float)
    T_trace   = raw[T_col].to_numpy(dtype=float) if T_col and T_col in raw else None

    return extract_rainflow_cycles(soc_trace, T_trace, model=model)


# ══════════════════════════════════════════════════════════════════════════════
#  Mode B – simulate SOC trace
# ══════════════════════════════════════════════════════════════════════════════
def simulate_soc_trace(
        speed_ms:       float,
        soc_final:      float,
        n_missions:     int   = 500,
        T_celsius:      float = 25.0,
        recharge_level: float = 1.0,
        recharge_noise: float = 0.02,
        pts_discharge:  int   = 100,
        pts_charge:     int   = 40,
        seed:           int   = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Synthesise a multi-mission SOC trace for Rainflow analysis.

    Parameters
    ----------
    speed_ms       : rover speed (m/s), used only for temperature model.
    soc_final      : residual SOC at end of each 10-km mission (fraction).
    n_missions     : number of missions to simulate.
    T_celsius      : nominal battery temperature (°C).
    recharge_level : mean SOC after recharge (1.0 = full, e.g. 0.95 = 95%).
    recharge_noise : std of recharge level variation (captures partial charges).
    pts_discharge  : interpolation points per discharge leg.
    pts_charge     : interpolation points per recharge leg.
    seed           : random seed for reproducibility.

    Returns
    -------
    soc_arr   : SOC trace  (1-D array)
    T_arr     : Temperature trace  (1-D array, same length)
    """
    rng = np.random.default_rng(seed)
    soc_arr, T_arr = [1.0], [T_celsius]
    dod_mission = 1.0 - soc_final

    for _ in range(n_missions):
        soc_start = soc_arr[-1]
        soc_end   = max(soc_start - dod_mission, 0.05)

        # discharge leg
        soc_leg = np.linspace(soc_start, soc_end, pts_discharge)
        soc_arr.extend(soc_leg[1:].tolist())
        T_arr.extend([T_celsius] * (pts_discharge - 1))

        # recharge leg (slight variability)
        soc_top = float(np.clip(
            rng.normal(recharge_level, recharge_noise), 0.5, 1.0))
        soc_leg = np.linspace(soc_end, soc_top, pts_charge)
        soc_arr.extend(soc_leg[1:].tolist())
        T_arr.extend([T_celsius] * (pts_charge - 1))

    return np.array(soc_arr), np.array(T_arr)


# ══════════════════════════════════════════════════════════════════════════════
#  Mode C – reproduce Fig 3 analysis
# ══════════════════════════════════════════════════════════════════════════════
def reproduce_fig3(
        speeds:    Sequence[float]  = (1.0, 2.0, 3.0, 4.0, 5.0),
        soc_finals: Sequence[float] = (53.5, 72.0, 75.1, 69.6, 65.8),
        missions_paper: Sequence[int] = (720, 975, 1040, 1180, 1105),
        n_sim:   int   = 500,
        T_fixed: float = 25.0,
        plot:    bool  = True,
        save_csv: str | None = None,
) -> pd.DataFrame:
    """
    Run the Rainflow analysis for each speed and compare with
    the observed mission counts from Fig 3(b).

    Returns a summary DataFrame with columns:
        speed, dod, fd_per_mission_rf, fd_per_mission_true,
        missions_pred_rf, missions_true,
        dod_mean_rf, S_delta_norm, fd_norm_data
    """
    SOH_EOL   = 0.80
    fd_eol    = -np.log(SOH_EOL)   # = 0.2231

    records = []
    for v, soc_f_pct, m_true in zip(speeds, soc_finals, missions_paper):
        soc_f = soc_f_pct / 100.0

        # simulate 500-mission SOC trace
        soc_arr, T_arr = simulate_soc_trace(
            speed_ms   = v,
            soc_final  = soc_f,
            n_missions = n_sim,
            T_celsius  = T_fixed,
        )

        # rainflow cycle extraction
        df_cyc = extract_rainflow_cycles(soc_arr, T_arr)
        fd_sim = df_cyc["fd_cycle"].sum()
        fd_per_mission_rf = fd_sim / n_sim

        # back-calculated true degradation rate
        fd_per_mission_true = fd_eol / m_true

        # predicted missions from simulation
        missions_pred = fd_eol / fd_per_mission_rf if fd_per_mission_rf > 0 else np.inf

        records.append({
            "speed":               v,
            "dod_mission":         1.0 - soc_f,
            "fd_per_mission_rf":   fd_per_mission_rf,
            "fd_per_mission_true": fd_per_mission_true,
            "missions_pred_rf":    missions_pred,
            "missions_true":       m_true,
            "dod_mean_rf":         df_cyc["avg_soc"].mean() if not df_cyc.empty else np.nan,
            "n_cycles_per_mission": df_cyc["count"].sum() / n_sim,
        })

    df = pd.DataFrame(records)

    # normalise to 3.0 m/s
    ref_row = df[df["speed"] == 3.0].iloc[0]
    df["S_delta_norm"]  = df["dod_mission"].apply(S_delta) / S_delta(ref_row["dod_mission"])
    df["fd_norm_data"]  = df["fd_per_mission_true"] / ref_row["fd_per_mission_true"]
    df["fd_norm_rf"]    = df["fd_per_mission_rf"]   / ref_row["fd_per_mission_rf"]

    print("\n" + "="*72)
    print("Rainflow cycle analysis — comparison with Fig 3(b)")
    print("="*72)
    print(df[["speed","dod_mission","fd_per_mission_rf","fd_per_mission_true",
              "missions_pred_rf","missions_true","fd_norm_rf","fd_norm_data"]].to_string(index=False))

    if save_csv:
        df.to_csv(save_csv, index=False)
        print(f"\nSaved → {save_csv}")

    if plot:
        _plot_fig3_c(df)

    return df


def _plot_fig3_c(df: pd.DataFrame) -> None:
    """Reproduce Panel (c) using Rainflow-derived fd alongside S_δ."""
    plt.rc("font", family="Times New Roman", size=14)

    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    x  = np.arange(len(df))
    w  = 0.32
    C_fd = "#1f4e79"
    C_Sd = "#059669"

    ax.bar(x - w/2, df["fd_norm_data"],  width=w, color=C_fd, alpha=0.88,
           label=r"$f_d$ (data, Eq. 7)")
    ax.bar(x + w/2, df["S_delta_norm"], width=w, color=C_Sd, alpha=0.88,
           label=r"$S_{\delta}$ (DOD factor, Eq. 8b)")

    ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--",
               alpha=0.55, zorder=2)
    ax.text(len(df) - 0.48, 1.03, "ref", fontsize=9,
            va="bottom", ha="right", color="gray", style="italic")

    # mark optimum
    opt = df["fd_norm_data"].idxmin()
    ox  = df.index.get_loc(opt)
    ax.annotate("", xy=(ox - w/2, df["fd_norm_data"].iloc[ox] + 0.02),
                xytext=(ox - w/2, df["fd_norm_data"].iloc[ox] + 0.18),
                arrowprops=dict(arrowstyle="-|>", color=C_fd,
                                lw=1.3, mutation_scale=10))
    ax.text(ox - w/2, df["fd_norm_data"].iloc[ox] + 0.20,
            "min", ha='center', va='bottom', fontsize=9, color=C_fd)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.1f}" for v in df["speed"]])
    ax.set_xlabel("Speed (m/s)")
    ax.set_ylabel("Normalised\ndegradation factor")
    ax.set_ylim(0.0, max(df["fd_norm_data"].max(), df["S_delta_norm"].max()) + 0.45)
    ax.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0])
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=1)
    ax.legend(fontsize=10, loc="upper right", framealpha=0.92,
              handlelength=1.0, borderpad=0.4, labelspacing=0.3)
    ax.set_title("(c) Per-mission degradation & DOD stress factor\n"
                 "(normalised to 3.0 m/s)", fontsize=11, pad=8)

    plt.tight_layout()
    plt.savefig("fig3c_rainflow.pdf", bbox_inches="tight")
    print("Saved → fig3c_rainflow.pdf")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(
        description="Extract Rainflow cycle data from battery SOC traces.")
    parser.add_argument("--mode", choices=["file", "simulate", "fig3"],
                        default="fig3",
                        help="file: read from CSV/NPY  |  "
                             "simulate: generate synthetic trace  |  "
                             "fig3: reproduce paper Fig 3 analysis (default)")
    parser.add_argument("--model", choices=["full", "without_dod", "without_avg_soc", "without_temperature"],
                        default="full",
                        help="Degradation model variant to use for the stress factors")
    parser.add_argument("--input",   type=str, default=None,
                        help="[file mode] path to SOC CSV/NPY file")
    parser.add_argument("--soc-col", type=str, default="soc",
                        help="[file mode] column name for SOC  (default: soc)")
    parser.add_argument("--T-col",   type=str, default=None,
                        help="[file mode] column name for temperature  (optional)")
    parser.add_argument("--speed",   type=float, default=3.0,
                        help="[simulate mode] rover speed in m/s  (default: 3.0)")
    parser.add_argument("--soc-final", type=float, default=75.1,
                        help="[simulate mode] residual SOC %%  (default: 75.1)")
    parser.add_argument("--n-missions", type=int, default=500,
                        help="[simulate] number of missions to simulate  (default: 500)")
    parser.add_argument("--save-csv", type=str, default=None,
                        help="Save cycle DataFrame to this CSV path")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip plotting")
    args = parser.parse_args()

    # ── MODE: fig3 ──────────────────────────────────────────────
    if args.mode == "fig3":
        df_summary = reproduce_fig3(plot=not args.no_plot,
                                     save_csv=args.save_csv)

    # ── MODE: file ──────────────────────────────────────────────
    elif args.mode == "file":
        if not args.input:
            parser.error("--input is required for file mode")
        df_cycles = cycles_from_file(args.input,
                                     soc_col=args.soc_col,
                                     T_col=args.T_col)
        if not df_cycles.empty:
            df_cycles = df_cycles.assign(model=args.model)
            df_cycles["S_sigma"], df_cycles["S_delta"], df_cycles["S_T"] = zip(
                *df_cycles.apply(
                    lambda r: stress_factors(r["dod"], r["avg_soc"], r["T_avg"], model=args.model),
                    axis=1,
                )
            )
            df_cycles["fd_cycle"] = df_cycles["count"] * df_cycles["S_sigma"] * df_cycles["S_delta"] * df_cycles["S_T"]
        print(f"\nExtracted {len(df_cycles)} cycles for model '{args.model}':")
        print(df_cycles.head(10).to_string(index=False))
        stats = summarise_cycles(df_cycles)
        print("\nSummary:")
        for k, v in stats.items():
            print(f"  {k:25s}: {v:.6g}")
        if args.save_csv:
            df_cycles.to_csv(args.save_csv, index=False)
            print(f"Saved → {args.save_csv}")

    # ── MODE: simulate ──────────────────────────────────────────
    elif args.mode == "simulate":
        soc_arr, T_arr = simulate_soc_trace(
            speed_ms   = args.speed,
            soc_final  = args.soc_final / 100.0,
            n_missions = args.n_missions,
        )
        df_cycles = extract_rainflow_cycles(soc_arr, T_arr, model=args.model)
        stats = summarise_cycles(df_cycles)
        fd_per_mission = stats["fd_total"] / args.n_missions
        missions_eol   = stats["fd_to_eol"] / fd_per_mission

        print(f"\nSimulated {args.n_missions} missions at {args.speed} m/s")
        print(f"  Model                    : {args.model}")
        print(f"  Residual SOC per mission : {args.soc_final:.1f}%")
        print(f"  Cycles extracted (RF)    : {len(df_cycles)}")
        print(f"  fd per mission           : {fd_per_mission:.10f}")
        print(f"  Predicted missions to EOL: {missions_eol:.0f}")
        print(f"\nFirst 10 cycles:")
        print(df_cycles.head(10)[["dod","avg_soc","count","T_avg",
                                   "S_sigma","S_delta","S_T","fd_cycle"]]
                                  .to_string(index=False))
        if args.save_csv:
            df_cycles.to_csv(args.save_csv, index=False)
            print(f"\nSaved → {args.save_csv}")