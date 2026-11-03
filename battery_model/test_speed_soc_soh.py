import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from battery_model import Battery2RCECM


def get_power(speed_mps: float) -> float:
    """Estimate the required propulsion power at a constant speed.

    The model is intentionally simple and only meant to compare the battery's
    final SOC/SOH after a fixed 10 km trip at different speeds.
    """
    if speed_mps <= 0:
        raise ValueError('speed_mps must be positive')
    # Rough vehicle power model in watts. Units here are m/s.
    power = 1.14005 * speed_mps + 0.10893 * speed_mps ** 3 + 1.1625
    power = power + 5.0 + speed_mps
    return float(power)


def simulate_10km(speed_mps: float, soh_init: float = 1.0, dt_s: float = 1.0) -> tuple[float, float]:
    """Run a constant-speed battery discharge over 10 km and return (final_soc, final_soh)."""
    distance_m = 10_000.0
    duration_s = distance_m / speed_mps
    battery = Battery2RCECM(
        capacity_ah=5.0,
        nominal_voltage=14.8,
        soc_init=1.0,
        soh_init=soh_init,
        temp_c=25.0,
        aging_enabled=True,
        max_substep_s=0.1,
    )

    elapsed = 0.0
    while elapsed < duration_s - 1e-9:
        remaining = duration_s - elapsed
        step_dt = min(dt_s, remaining)
        power_w = get_power(speed_mps)
        try:
            soc, soh = battery.step(power_w=power_w, dt=step_dt)
        except Exception:
            raise RuntimeError(f'Battery cannot sustain power {power_w:.1f} W at {speed_mps:.2f} m/s')
        elapsed += step_dt

    return float(battery.soc), float(battery.soh)


def plot_results(speeds_mps: list[float], final_soc: list[float], final_soh: list[float]) -> None:
    out_dir = Path(__file__).resolve().parent
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(speeds_mps, final_soc, 'o-', color='tab:blue', label='final SOC')
    ax1.set_xlabel('Speed (m/s)')
    ax1.set_ylabel('Final SOC', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, linestyle='--', alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(speeds_mps, final_soh, 's--', color='tab:orange', label='final SOH')
    ax2.set_ylabel('Final SOH', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    plt.title('10 km final SOC/SOH vs speed')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

    fig.tight_layout()
    fig.savefig(out_dir / 'speed_vs_final_soc_soh.png', dpi=200)
    plt.close(fig)

def main() -> None:
    # evaluate a grid of speed and initial SOH values
    speeds_mps = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    soh_values = [0.85, 0.90, 0.95, 1.0]
    soc_matrix: list[list[float]] = []
    soh_matrix: list[list[float]] = []

    print('speed_mps, soh_init, final_soc, final_soh')
    for v in speeds_mps:
        row_soc = []
        row_soh = []
        for soh in soh_values:
            try:
                soc, soh_final = simulate_10km(v, soh_init=soh)
            except RuntimeError:
                soc = float('nan')
                soh_final = float('nan')
            row_soc.append(soc)
            row_soh.append(soh_final)
            print(f'{v:.2f},{soh:.2f},{soc if math.isfinite(soc) else "N/A"},{soh_final if math.isfinite(soh_final) else "N/A"}')
        soc_matrix.append(row_soc)
        soh_matrix.append(row_soh)

    # Separate plot for final SOH only
    fig_soh, ax_soh = plt.subplots(figsize=(8, 5))
    for idx, soh in enumerate(soh_values):
        soh_values_curve = [soh_matrix[i][idx] for i in range(len(speeds_mps)) if math.isfinite(soh_matrix[i][idx])]
        xs = [speeds_mps[i] for i in range(len(speeds_mps)) if math.isfinite(soh_matrix[i][idx])]
        if xs:
            ax_soh.plot(xs, soh_values_curve, 'o-', label=f'Initial SOH={soh:.2f}')
    ax_soh.set_xlabel('Speed (m/s)')
    ax_soh.set_ylabel('Final SOH')
    ax_soh.set_title('Final SOH vs speed at different initial SOH')
    ax_soh.grid(True, linestyle='--', alpha=0.3)
    ax_soh.legend(loc='best')
    fig_soh.tight_layout()
    fig_soh.savefig(Path(__file__).resolve().parent / 'final_soh_vs_speed_by_soh.png', dpi=200)
    plt.close(fig_soh)

    print('\nSaved figure: final_soh_vs_speed_by_soh.png')


if __name__ == '__main__':
    main()
