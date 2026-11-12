"""Evaluate policies on the test grid; print / save summaries; load DARO."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Dict, Optional

import numpy as np

from .test_env import test_suite, make_test_env
from .metrics import run_episode, aggregate


def evaluate_policies(policies: Dict[str, object], config: dict,
                      n_episodes: Optional[int] = None,
                      verbose: bool = True) -> dict:
    """results[policy] = {'overall': {...}, 'scenarios': {name: {...}}}."""
    n_episodes = n_episodes or config['test_episodes']
    suite = test_suite(config)
    results: Dict[str, dict] = {}
    for pol_name, pol in policies.items():
        per_scen, all_eps = {}, []
        for k, scen in enumerate(suite):
            env = make_test_env(config, scen)
            eps = [run_episode(env, pol, seed=config['seed'] + 1000 * k + i,
                               charge_c_rate=config['charge_c_rate'])
                   for i in range(n_episodes)]
            per_scen[scen.name] = dict(asdict(scen), **aggregate(eps))
            all_eps.extend(eps)
        results[pol_name] = {'overall': aggregate(all_eps), 'scenarios': per_scen}
    if verbose:
        print_summary(results, config)
    return results


COLS = [('return', 'Return', '{:8.2f}'), ('energy_kj', 'E[kJ]', '{:6.2f}'),
        ('wh_per_km', 'Wh/km', '{:6.2f}'), ('mission_time_s', 'T[s]', '{:6.0f}'),
        ('completed', 'Done%', '{:5.0%}'), ('qos_viol_pct', 'QoSv%', '{:6.1f}'),
        ('stress_proxy', 'Stress', '{:7.1f}'), ('delta_soc', 'dSOC', '{:6.3f}'),
        ('delta_soh_ppm', 'dSOH[ppm]', '{:9.1f}')]


def format_row(name, m, cols=COLS):
    return f"{name:<10}" + ''.join(' ' + fmt.format(m[k]).rjust(max(len(h), 6))
                                   for k, h, fmt in cols)


def format_header(first='Policy', cols=COLS):
    return f"{first:<10}" + ''.join(' ' + h.rjust(max(len(h), 6)) for _, h, _ in cols)


def print_summary(results: dict, config: dict):
    print(f"\n=== Overall, mission {config['mission_distance_m']:.0f} m "
          f"(mean over all scenarios and seeds) ===")
    print(format_header())
    for name, r in results.items():
        print(format_row(name, r['overall']))
    for key, label, levels in (('soh', 'SOH', config['test_soh_levels']),
                               ('soc_init', 'initial SOC', config['test_soc_levels'])):
        print(f'\n=== By {label} ===')
        for lvl in levels:
            print(f'-- {label} = {lvl:.2f}')
            for name, r in results.items():
                rows = [s for s in r['scenarios'].values() if abs(s[key] - lvl) < 1e-9]
                print(format_row(name, {k: float(np.mean([s[k] for s in rows]))
                                        for k, _, _ in COLS}))
    print()


def save_results(results: dict, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f'Results saved -> {path}')


def load_daro(ckpt_dir: str, config: dict):
    """Build a SACAgent matching the training config and load weights.

    The observation variant stored with the checkpoint (obs_variants.py) is
    applied automatically: RL-SOC-only / RL-SOH-only agents get the masked obs.
    """
    import sac_training as st
    from obs_variants import MaskedPolicy, load_variant
    train_cfg = dict(st.DEFAULT_CONFIG, control_dt=config['control_dt'])
    env = st.build_env(train_cfg)
    agent = st.build_agent(train_cfg, env)
    agent.load(ckpt_dir, eval_only=True)
    variant = load_variant(ckpt_dir)
    return agent if variant == 'full' else MaskedPolicy(agent, variant, env.obs_dim)


def rl_label(ckpt_dir: str) -> str:
    """Method name from the checkpoint's variant file, e.g. 'DARO', 'RL-SOC-only',
    'DARO w/o D_d' (reward variant no_stress), 'RL-SOH-only w/o E_d'."""
    from obs_variants import label, read_variants
    v = read_variants(ckpt_dir)
    return label(v['obs'], v['reward'])


def load_rl_policies(ckpt_dirs, config: dict):
    """Load several RL checkpoints, grouped by method.

    Returns (policies, groups): policies[name] = agent; groups[label] = [names].
    With one checkpoint per method the name is the label ('DARO'); with several
    seeds the names are '<label> s<k>' (k from the 'seed_<k>' directory name).
    """
    import os
    by_label = {}
    for ck in ckpt_dirs:
        if not os.path.isfile(os.path.join(ck, 'daro_checkpoint.pt')):
            raise SystemExit(f'No daro_checkpoint.pt in: {ck}. '
                             'Note: "[...]" in usage text means optional; do not type the brackets.')
        by_label.setdefault(rl_label(ck), []).append(ck)
    policies, groups = {}, {}
    for label, cks in by_label.items():
        for ck in cks:
            seed = os.path.basename(os.path.normpath(ck)).replace('seed_', '')
            name = label if len(cks) == 1 else f'{label} s{seed}'
            if name in policies:
                raise SystemExit(f'Duplicate policy name {name!r} ({ck}).')
            policies[name] = load_daro(ck, config)
            groups.setdefault(label, []).append(name)
    return policies, groups
