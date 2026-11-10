"""python -m evaluation [--ckpt DIR | --baselines-only] [--distance M] ..."""

import argparse
import os

from . import (EVAL_CONFIG, make_baselines, evaluate_policies, save_results,
               load_daro, lifetime_from_results)


def main():
    ap = argparse.ArgumentParser(description='DARO test-environment evaluation')
    ap.add_argument('--ckpt', type=str, default=None, help='DARO checkpoint directory')
    ap.add_argument('--baselines-only', action='store_true')
    ap.add_argument('--episodes', type=int, default=None, help='Seeds per scenario')
    ap.add_argument('--distance', type=float, default=None, help='Mission distance [m]')
    ap.add_argument('--control-dt', type=float, default=None)
    ap.add_argument('--out', type=str, default=None)
    args = ap.parse_args()

    config = dict(EVAL_CONFIG)
    if args.control_dt is not None:
        config['control_dt'] = args.control_dt
    if args.distance is not None:
        config['mission_distance_m'] = args.distance
    policies = {}
    if not args.baselines_only:
        if not args.ckpt:
            raise SystemExit('Pass --ckpt <dir> or --baselines-only.')
        policies['DARO'] = load_daro(args.ckpt, config)
    policies.update(make_baselines(config))
    results = evaluate_policies(policies, config, n_episodes=args.episodes)
    life = lifetime_from_results(results, config)
    print('=== Lifetime (full charge per mission, extrapolated Xu fade) ===')
    for pol, per_c in life.items():
        print(f"{pol:<10} " + '  '.join(f"C={c:g}: {v['missions_to_eol']:7.0f} missions "
                                       f"/ {v['km_to_eol']:7.0f} km" for c, v in per_c.items()))
    save_results({'config': config, 'results': results, 'lifetime': life},
                 args.out or os.path.join(config['log_dir'], 'test_results.json'))


if __name__ == '__main__':
    main()
