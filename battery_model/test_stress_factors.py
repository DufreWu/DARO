# Minimal regression tests for correct Xu stress-factor placement.
import math
from battery_model import Battery2RCECM, _xu_fd_halfcycle, _DEFAULT_AGING_PARAMS
kw = _DEFAULT_AGING_PARAMS
f = lambda dod, avg: _xu_fd_halfcycle(dod, avg, 298.15, **kw)
assert f(0, 0.7) == 0.0
assert f(0.05, 0.7) < f(0.2, 0.7) < f(0.4, 0.7), 'DoD aging must increase'
assert f(0.2, 0.4) < f(0.2, 0.8), 'Average SOC stress must increase'
assert f(1e-7, 0.7) < f(1e-4, 0.7), 'No discontinuity at zero DoD'
from test_speed_mission import run_one_mission
battery = Battery2RCECM()
soc_end, soh_end = run_one_mission(battery, 3.0, distance_m=100.0)
assert 0.0 < soc_end < 1.0, 'Returned SOC should be pre-recharge'
assert battery.soc == 1.0, 'SOC should reset after simulated recharge'
assert soh_end < 1.0
print('PASS: Xu stress-factor placement and pre-recharge SOC logging')
