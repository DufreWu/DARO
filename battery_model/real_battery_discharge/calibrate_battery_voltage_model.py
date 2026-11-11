"""Empirical fitted DISCHARGE voltage curves based on screenshot (not measured OCV).

SOC inferred as 1 - t/t_end, assuming constant current, full usable SOC range.
Aged SOH of 0.85 is a demonstration mapping only; aged SOH was not given.
The instantaneous voltage includes internal-resistance/polarization drop at the original, unknown load.
Use rested measurements for an actual OCV lookup in an ECM.
"""
import numpy as np

FRESH_COEFF = [15.061061, 0.0, 0.995533, 1.18033, 0.881677]
AGED_COEFF = [15.008879, 0.077644, 1.31415, 0.976887, 0.855835]

def _curve(soc, coeff):
    s = np.clip(np.asarray(soc, dtype=float), 0.0, 1.0)
    x = 1.0 - s
    v0, a1, a3, a8, a25 = coeff
    return v0 - a1*x - a3*x**3 - a8*x**8 - a25*x**25

def v_discharge(soc, soh=1.0, soh_aged_ref=0.85):
    """Empirical terminal-voltage shape; interpolated ageing is illustrative.

    WARNING: this does NOT represent OCV and must not be placed into
    V_term = OCV - I*R without compensating for the load already in the fit.
    """
    if not (0.0 < soh_aged_ref < 1.0):
        raise ValueError('soh_aged_ref must be between 0 and 1')
    w = np.clip((1.0 - soh)/(1.0 - soh_aged_ref), 0.0, 1.0)
    voltage = (1.0 - w)*_curve(soc, FRESH_COEFF) + w*_curve(soc, AGED_COEFF)
    return float(voltage) if np.ndim(voltage)==0 else voltage

if __name__ == '__main__':
    for soc in (1.0, 0.75, 0.5, 0.25, 0.1, 0.0):
        print(f'SOC={soc:4.2f}  fresh={v_discharge(soc):5.2f} V  aged={v_discharge(soc,0.85):5.2f} V')
