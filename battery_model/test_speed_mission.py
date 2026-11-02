import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from battery_model import Battery2RCECM


DEFAULT_SPEEDS = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]


def get_power(speed_mps: float) -> float:
    """Rough propulsion power model in watts for a constant-speed mission."""
    if speed_mps <= 0:
        raise ValueError('speed_mps must be positive')
    power = 1.14005 * speed_mps + 0.10893 * speed_mps ** 3 + 1.1625
    power = power + 5.0 + speed_mps
    return float(power)


def run_one_mission(
    battery: Battery2RCECM,
    speed_mps: float,
    distance_m: float = 10_000.0,
    dt_s: float = 1.0,
) -> tuple[float, float]:
    """Run one fixed-distance mission and return (final_soc, final_soh)."""
    duration_s = distance_m / speed_mps
    elapsed = 0.0
    while elapsed < duration_s - 1e-9:
        remaining = duration_s - elapsed
        step_dt = min(dt_s, remaining)
        try:
            battery.step(power_w=get_power(speed_mps), dt=step_dt)
        except Exception as exc:
            raise RuntimeError(
                f'Battery cannot sustain power at {speed_mps:.2f} m/s: {exc}'
            ) from exc
        elapsed += step_dt

    # Commit the half-cycle aging effect before recharging.
    battery.cycle_end()

    # Recharge back to full SOC while preserving the degraded SOH.
    battery.soc = 1.0
    battery.v1 = 0.0
    battery.v2 = 0.0
    battery._soc_cycle_start = 1.0

    return float(battery.soc), float(battery.soh)


def missions_to_eol(
    speed_mps: float,
    eol_soh: float = 0.8,
    distance_m: float = 10_000.0,
    dt_s: float = 1.0,
    max_missions: int = 20000,
    initial_soh: float = 1.0,
) -> dict:
    """Simulate repeated missions until the battery reaches EOL.

    After each mission, the battery is recharged to 100% SOC and the degraded
    SOH is retained for the next mission. EOL is defined by SOH <= eol_soh.
    """
    if not (0.5 <= eol_soh <= 1.0):
        raise ValueError('eol_soh must be within [0.5, 1.0]')

    battery = Battery2RCECM(
        capacity_ah=5.0,
        nominal_voltage=14.8,
        soc_init=1.0,
        soh_init=initial_soh,
        temp_c=25.0,
        aging_enabled=True,
        max_substep_s=0.1,
    )

    mission_count = 0
    while battery.soh > eol_soh:
        if mission_count >= max_missions:
            raise RuntimeError(
                f'Exceeded max_missions={max_missions} before reaching EOL at {speed_mps:.2f} m/s'
            )
        mission_count += 1
        _, _ = run_one_mission(battery, speed_mps, distance_m=distance_m, dt_s=dt_s)

        # print(battery.soc, battery.soh, eol_soh)

    return {
        'speed_mps': float(speed_mps),
        'missions_to_eol': mission_count,
        'final_soc': float(battery.soc),
        'final_soh': float(battery.soh),
        'eol_soh': float(eol_soh),
        'eol_reached': True,
    }


def plot_results(speeds: list[float], missions: list[int]) -> Path:
    out_dir = Path(__file__).resolve().parent
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(speeds, missions, 'o-', color='tab:green')
    ax.set_xlabel('Speed (m/s)')
    ax.set_ylabel('Mission count to EOL')
    ax.set_title('Mission count to EOL vs speed')
    ax.grid(True, linestyle='--', alpha=0.3)
    fig.tight_layout()
    out_path = out_dir / 'mission_count_to_eol_vs_speed.png'
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description='Run mission-count-to-EOL simulations for different speeds.')
    parser.add_argument('--speeds', type=str, default='1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0')
    parser.add_argument('--eol-soh', type=float, default=0.8)
    parser.add_argument('--distance-m', type=float, default=10_000.0)
    parser.add_argument('--dt-s', type=float, default=1.0)
    parser.add_argument('--max-missions', type=int, default=20000)
    parser.add_argument('--initial-soh', type=float, default=1.0)
    parser.add_argument('--plot', action='store_true', help='Plot speed vs missions to EOL')
    args = parser.parse_args()

    speeds = [float(v.strip()) for v in args.speeds.split(',') if v.strip()]
    if not speeds:
        raise ValueError('No speed values provided')

    print('speed_mps, missions_to_eol, final_soc, final_soh')
    results = []
    for speed in speeds:
        if speed <= 0:
            raise ValueError(f'All speeds must be positive; got {speed}')
        result = missions_to_eol(
            speed_mps=speed,
            eol_soh=args.eol_soh,
            distance_m=args.distance_m,
            dt_s=args.dt_s,
            max_missions=args.max_missions,
            initial_soh=args.initial_soh,
        )
        results.append(result)
        print(
            f'{result["speed_mps"]:.2f}, '
            f'{result["missions_to_eol"]}, '
            f'{result["final_soc"]:.6f}, '
            f'{result["final_soh"]:.12f}'
        )

    if args.plot and results:
        fig_path = plot_results(
            [r['speed_mps'] for r in results],
            [r['missions_to_eol'] for r in results],
        )
        print(f'\nSaved figure: {fig_path.name}')


if __name__ == '__main__':
    main()

    # 1.00, 16128, 1.000000, 0.799996445453