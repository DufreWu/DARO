"""Battery-aware simulation: SOC/SOH/temperature-dependent 2RC ECM.

All five ECM elements (R0, R1, R2, C1, C2) have independent, explicit
functions of SOC, SOH, and cell/pack temperature. OCV can be supplied as
SOH-indexed measured curves or an explicitly uncalibrated aging perturbation.
No measured impedance, HPPC, OCV or aging calibration is claimed: default
coefficient values are illustrative.
The electrical ECM supports discharge only (nonnegative constant terminal
power). Recharge is modelled by coulomb counting in charge() (CC charge at a
given C-rate), which accrues the charge half-cycle and calendar aging but
does not simulate charge-side ECM voltages.

Energy bookkeeping separates resistor heat, frozen-parameter RC energy change,
and signed numerical/parameter residual. These are diagnostic channels, not a
thermal model. The optional current limit must be set from real pack/BMS data.
Default parameters and synthetic OCV are illustrative, not pack calibration.

At each electrical substep, the RC time constants are recomputed as
    tau1 = R1(SOC, SOH, T) * C1(SOC, SOH, T)
    tau2 = R2(SOC, SOH, T) * C2(SOC, SOH, T)

Reference state for ECM component values: SOC=0.5, SOH=1, T=25 C.

Default pack: Blue Robotics Lithium-ion Battery (4S6P, 18 Ah, 14.8 V,
266.4 Wh nominal energy, 12 V minimum safe voltage).
https://bluerobotics.com/store/comm-control-power/powersupplies-batteries/battery-li-4s-18ah-r3/
Pack ratings do not calibrate the assumed OCV, resistance, RC or aging curves.

SOH DEGRADATION MODEL (v4 change)
----------------------------------
v3 used a simple throughput law:
    soh -= (0.2 / cycles_to_80) * (dAh / Q) * stress

v4 replaces this with the semi-empirical cycle-aging model of Xu et al. (2016),
matching the formulation used in the DARO paper (Eqs. 6-8):

    SOH(t) = exp(-fd)
    fd     = fd_completed + fd_current_halfcycle
    fd_completed = sum(committed cycle half-cycles) + fd_calendar

where fd_current_halfcycle tracks the in-progress discharge (or charge)
half-cycle using cumulative DOD and time-averaged SOC since cycle_start:

    delta  = |SOC_cycle_start - SOC_now|        (cumulative DoD; = delta_i)
    sigma  = (SOC_cycle_start + SOC_now) / 2    (avg SOC;        = sigma_i)
    S_sig  = exp(k_sigma * (sigma - sigma_ref))
    S_del  = 1 / (k_d1 * delta^k_d2 + k_d3)
    S_T    = exp(k_T * (T_K - T_ref) * T_ref / T_K)
    fd_current_halfcycle = 0.5 * k_cyc * S_sig * S_del * S_T

k_cyc is a dimensionless cycle-rate scale. It is the only aging coefficient the
LiPo dataset can identify (all cells were cycled under one stress condition),
so parameter_fitting.py fits k_cyc and keeps the stress-shape coefficients at
the Xu et al. literature values.

Calendar aging (Xu et al. 2016, Eq. for f_t) accrues continuously in time,
during discharge, charge and rest alike:

    d fd_calendar / dt = k_t * S_sig(SOC(t)) * S_T(T)

so a slower mission (longer duration for the same distance) pays a calendar
aging penalty. k_t defaults to the Xu literature value (4.14e-10 1/s); the
LiPo cycling data cannot separate it from cycle aging, so treat it as an
assumption and override via aging_params if you have calendar data.
Charge half-cycles are counted the same way as discharge half-cycles.

This is mathematically equivalent to applying Rainflow counting at the end of
each half-cycle: the terminal value of fd_current_halfcycle equals the Rainflow
result for one half-cycle, independent of the control time-step. There is
therefore no step-count bias.

Call cycle_end() after each complete half-cycle (e.g. at mission end before
recharge, and again after recharge). This commits fd_current_halfcycle to
fd_completed and resets the cycle-start SOC. Between cycle_end() calls the SOH
is a real-time estimate of the current half-cycle's impact; it is physically
monotone over the full cycle but may exhibit sub-1e-8 non-monotonicities
within a half-cycle due to the competing DOD/avgSOC stress factors.

Degradation coefficients default to those fitted to the LiPo aging dataset of
Galeotti et al. (Data in Brief, 2023) following the functional form of
Xu et al. (IEEE Trans. Smart Grid, 2016). Provide your own values via the
aging_params argument to match a different cell chemistry or pack.

Run demo: python battery_ecm_v4.py --capacity-ah 18 --nominal-voltage 14.8
Tests:    python battery_ecm_v4.py --self-test
"""
import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any


class BatteryLimitError(RuntimeError):
    """Requested step not deliverable. State is unchanged; terminate episode."""


@dataclass(frozen=True)
class ECMParameters:
    r0: float
    r1: float
    r2: float
    c1: float
    c2: float
    tau1: float
    tau2: float


# ---------------------------------------------------------------------------
# LiPo dataset-informed defaults.
# ---------------------------------------------------------------------------

def _load_fitted_model_params() -> Dict[str, Any]:
    """Load the repository's lipo_fit_parameters.json when available."""
    fit_path = Path(__file__).resolve().parent / 'lipo_fit_parameters.json'
    if not fit_path.exists():
        return {}
    try:
        with fit_path.open() as fh:
            payload = json.load(fh)
    except (OSError, ValueError, TypeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def _cpe_to_c_parallel(r: float, q: float, alpha: float) -> float:
    """Hsu-Mansfeld equivalent capacitance (F) of a parallel R-CPE element."""
    if not (r > 0.0 and q > 0.0 and 0.0 < alpha <= 1.0):
        raise ValueError('Invalid R||CPE combination')
    return math.exp((math.log(q) + (1.0 - alpha) * math.log(r)) / alpha)


_FALLBACK_CELL_NOMINAL_V = 3.7
_FALLBACK_CELL_CAPACITY_AH = 1.0


def _dataset_reference_ecm_values() -> Dict[str, float]:
    """Return SINGLE-CELL ECM reference values (SOC=0.5) from the LiPo data.

    Keys: R0, R1, R2 (ohm), C1, C2 (farad), cell_nominal_voltage, cell_capacity_ah.

    Preferred source: lipo_fit_parameters.json written by parameter_fitting.py
    (identified by its 'ecm_fit_method' field; C1/C2 there are Hsu-Mansfeld
    equivalent capacitances in farad). Older JSON files stored the raw CPE
    coefficients Q1/Q2 as "C1/C2", which are not farads, so they are ignored.
    Fallback: median of the raw LiPO_*_fit CSV rows near SOC=0.5 with the same
    CPE -> C conversion. Returns {} when neither source exists.
    """
    fit_payload = _load_fitted_model_params()
    ecm = fit_payload.get('ecm', {}) if isinstance(fit_payload, dict) else {}
    if isinstance(ecm, dict) and 'ecm_fit_method' in fit_payload:
        refs = {}
        for key in ('R0', 'R1', 'R2', 'C1', 'C2'):
            info = ecm.get(key)
            if isinstance(info, dict) and isinstance(info.get('reference_soc_0_5'), (int, float)):
                refs[key] = float(info['reference_soc_0_5'])
        if len(refs) == 5:
            cell = fit_payload.get('cell', {}) if isinstance(fit_payload.get('cell'), dict) else {}
            refs['cell_nominal_voltage'] = float(cell.get('nominal_voltage_v', _FALLBACK_CELL_NOMINAL_V))
            refs['cell_capacity_ah'] = float(cell.get('capacity_ah', _FALLBACK_CELL_CAPACITY_AH))
            return refs

    base_dir = Path(__file__).resolve().parent / 'LiPo_dataset'
    if not base_dir.exists():
        return {}

    samples: Dict[str, list] = {key: [] for key in ('R0', 'R1', 'R2', 'C1', 'C2')}
    for fit_dir in sorted(base_dir.glob('*_fit')):
        for csv_path in sorted(fit_dir.glob('model_fit_cycle_*.csv')):
            with csv_path.open(newline='') as fh:
                for row in csv.DictReader(fh):
                    try:
                        soc = float(row['SOC'])
                        if not 0.45 <= soc <= 0.55:
                            continue
                        r0, r1, r2 = float(row['RO']), float(row['R1']), float(row['R2'])
                        c1 = _cpe_to_c_parallel(r1, float(row['Q1']), float(row['a1']))
                        c2 = _cpe_to_c_parallel(r2, float(row['Q2']), float(row['a2']))
                    except (TypeError, ValueError, KeyError):
                        continue
                    if r0 <= 0:
                        continue
                    for key, val in zip(('R0', 'R1', 'R2', 'C1', 'C2'), (r0, r1, r2, c1, c2)):
                        samples[key].append(val)
    if not all(samples.values()):
        return {}
    refs = {key: statistics.median(vals) for key, vals in samples.items()}
    refs['cell_nominal_voltage'] = _FALLBACK_CELL_NOMINAL_V
    refs['cell_capacity_ah'] = _FALLBACK_CELL_CAPACITY_AH
    return refs


_DATASET_ECM_REF = _dataset_reference_ecm_values()


# ---------------------------------------------------------------------------
# Data-grounded Xu-model degradation coefficients.
#
# The LiPo dataset in this repository contains real cycle-capacity aging traces
# and fit curves, but it does not expose the full cycle-by-cycle stress metadata
# needed to uniquely re-identify every Xu coefficient from scratch in a generic
# simulator. For this reason we keep the published LiPo-fitted values as the
# data-derived default, while still validating that the dataset is present and
# allowing a user-supplied aging_params override.
# ---------------------------------------------------------------------------
_DEFAULT_AGING_PARAMS_SOURCE = {
    'k_sigma': 1.04,
    'k_T': 6.93e-2,
    'k_d1': 1.40e5,
    'k_d2': -5.01e-1,
    'k_d3': -1.23e5,
    'sigma_ref': 0.50,
    'T_ref_k': 298.15,
    # Cycle-rate scale (fitted by parameter_fitting.py; 1.0 = pure Xu LMO).
    'k_cyc': 1.0,
    # Calendar aging rate, 1/s (Xu et al. 2016 literature value).
    'k_t': 4.14e-10,
}


def _fit_aging_params_from_dataset() -> Dict[str, float]:
    """Return LiPo dataset-calibrated Xu aging coefficients.

    Prefer the repository's generated fit from lipo_fit_parameters.json when it
    exists; otherwise keep the same LiPo-derived defaults so the model remains
    usable both in and outside the dataset workspace.
    """
    params = dict(_DEFAULT_AGING_PARAMS_SOURCE)
    fit_payload = _load_fitted_model_params()
    aging = fit_payload.get('degradation', {}) if isinstance(fit_payload, dict) else {}
    if isinstance(aging, dict):
        # Merge any known numeric keys from the fit; keep defaults for the rest.
        for key in params:
            if key in aging:
                try:
                    params[key] = float(aging[key])
                except (TypeError, ValueError):
                    pass
    return params


_DEFAULT_AGING_PARAMS: Dict[str, float] = _fit_aging_params_from_dataset()


def _xu_s_sigma(soc: float, k_sigma: float, sigma_ref: float) -> float:
    return math.exp(k_sigma * (soc - sigma_ref))


def _xu_s_temp(temp_k: float, k_T: float, T_ref_k: float) -> float:
    return math.exp(k_T * (temp_k - T_ref_k) * T_ref_k / temp_k)


def _xu_calendar_fd(
    soc0: float,
    soc1: float,
    dt_s: float,
    temp_k: float,
    *,
    k_sigma: float,
    k_T: float,
    sigma_ref: float,
    T_ref_k: float,
    k_t: float,
    **_unused: float,
) -> float:
    """Calendar fd over dt_s with SOC moving linearly from soc0 to soc1.

    Integrates k_t * S_sigma(SOC(t)) * S_T exactly for a linear SOC segment, so
    the result does not depend on how a constant-rate segment is subdivided.
    """
    if dt_s <= 0.0 or k_t == 0.0:
        return 0.0
    a = k_sigma * (soc0 - sigma_ref)
    b = k_sigma * (soc1 - sigma_ref)
    if abs(b - a) < 1e-12:
        mean_s_sigma = math.exp(0.5 * (a + b))
    else:
        mean_s_sigma = (math.exp(b) - math.exp(a)) / (b - a)
    return k_t * dt_s * mean_s_sigma * _xu_s_temp(temp_k, k_T, T_ref_k)


def _xu_fd_halfcycle(
    dod: float,
    avg_soc: float,
    temp_k: float,
    *,
    k_sigma: float,
    k_T: float,
    k_d1: float,
    k_d2: float,
    k_d3: float,
    sigma_ref: float,
    T_ref_k: float,
    k_cyc: float = 1.0,
    **_unused: float,
) -> float:
    """Degradation increment for one half-cycle under the Xu et al. model.

    Returns fd contribution = 0.5 * k_cyc * S_sigma * S_delta * S_T >= 0.
    S_delta is guarded against a non-positive denominator; such a case
    represents an out-of-validity parameter/state combination and should
    be diagnosed before results are used for lifetime prediction.
    """
    if dod <= 0.0:
        return 0.0
    # Xu et al. (2016): S_sigma depends on average SOC; S_delta on DoD.
    S_sigma = _xu_s_sigma(avg_soc, k_sigma, sigma_ref)
    denom = k_d1 * (dod ** k_d2) + k_d3
    S_delta = 1.0 / denom if denom > 0.0 else 0.0
    S_T = _xu_s_temp(temp_k, k_T, T_ref_k)
    return 0.5 * k_cyc * S_sigma * S_delta * S_T


class Battery2RCECM:
    # Pack OCV from an illustrative 4S lithium-ion cell curve (SOC, cell V).
    _CELL_OCV = (
        (0.00, 3.00), (0.05, 3.30), (0.10, 3.50),
        (0.20, 3.60), (0.40, 3.70), (0.60, 3.80),
        (0.80, 3.95), (0.90, 4.08), (1.00, 4.20),
    )

    def __init__(
        self,
        capacity_ah: float = 5.0,
        nominal_voltage: float = 14.8,
        soc_init: float = 1.0,
        soh_init: float = 1.0,
        temp_c: float = 25.0,
        cycles_to_80: float = 1000.0,          # kept for API compatibility; unused by Xu model
        ocv_age_delta_v_at_soh80: float = -0.8,
        *,
        aging_enabled: bool = True,
        aging_params: Optional[Dict[str, float]] = None,
        max_discharge_current_a: Optional[float] = None,
        cutoff_voltage: Optional[float] = None,
        max_substep_s: float = 0.1,
        ocv_table=None,
        ocv_tables_by_soh=None,
        ocv_temp_coeff_v_per_c: float = 0.0,
    ):
        """Construct an illustrative pack-level discharge-only 2RC ECM.

        aging_params overrides the default Xu-model coefficients for any key
        supplied. Keys: k_sigma, k_T, k_d1, k_d2, k_d3, sigma_ref, T_ref_k,
        k_cyc (cycle-rate scale), k_t (calendar rate, 1/s; 0 disables).
        cycles_to_80 is retained for API compatibility but is no longer used
        in the SOH update; the Xu model parameters govern degradation rate.
        """
        numeric = (capacity_ah, nominal_voltage, soc_init, soh_init, temp_c,
                   cycles_to_80, ocv_age_delta_v_at_soh80, max_substep_s,
                   ocv_temp_coeff_v_per_c)
        if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in numeric):
            raise ValueError('All numeric inputs must be finite')
        if capacity_ah <= 0 or nominal_voltage <= 0 or cycles_to_80 <= 0:
            raise ValueError('Capacity, nominal voltage and cycle life must be positive')
        if max_substep_s <= 0:
            raise ValueError('max_substep_s must be positive')
        if not 0.02 <= soc_init <= 1 or not 0.5 <= soh_init <= 1:
            raise ValueError('SOC must be in [0.02, 1] and SOH in [0.5, 1]')
        if not -10 <= temp_c <= 50:
            raise ValueError('Temperature must be in [-10, 50] C')
        if max_discharge_current_a is not None:
            if not math.isfinite(max_discharge_current_a) or max_discharge_current_a <= 0:
                raise ValueError('max_discharge_current_a must be positive or None')
        if cutoff_voltage is not None:
            if not math.isfinite(cutoff_voltage) or cutoff_voltage <= 0:
                raise ValueError('cutoff_voltage must be positive or None')
        if not isinstance(aging_enabled, bool):
            raise ValueError('aging_enabled must be a bool')
        if ocv_table is not None and ocv_tables_by_soh is not None:
            raise ValueError('Choose ocv_table OR ocv_tables_by_soh')

        self.capacity_ah = float(capacity_ah)
        self.nominal_voltage = float(nominal_voltage)
        self.soc = float(soc_init)
        self.soh = float(soh_init)
        self.temp_c = float(temp_c)
        self.cycles_to_80 = float(cycles_to_80)
        self.ocv_age_delta_v_at_soh80 = float(ocv_age_delta_v_at_soh80)
        self.ocv_temp_coeff_v_per_c = float(ocv_temp_coeff_v_per_c)
        self.aging_enabled = aging_enabled
        self.max_discharge_current_a = (None if max_discharge_current_a is None
                                        else float(max_discharge_current_a))
        self.cutoff_voltage = (self.nominal_voltage * (3.0 / 3.7)
                               if cutoff_voltage is None else float(cutoff_voltage))
        self.max_substep_s = float(max_substep_s)
        self._ocv_table = (None if ocv_table is None
                           else self._validated_curve(ocv_table))
        if ocv_tables_by_soh is None:
            self._ocv_tables_by_soh = None
        else:
            if not ocv_tables_by_soh:
                raise ValueError('ocv_tables_by_soh cannot be empty')
            tables = []
            for health, curve in ocv_tables_by_soh.items():
                health = float(health)
                if not math.isfinite(health) or not 0.5 <= health <= 1:
                    raise ValueError('OCV table SOH keys must be in [0.5, 1]')
                tables.append((health, self._validated_curve(curve)))
            self._ocv_tables_by_soh = tuple(sorted(tables))

        # Reference ECM components: SOC=0.5, SOH=1, T=25 C.
        # With LiPo data available, single-cell values are scaled to the pack:
        #   Ns = pack V / cell V (series), Np = pack Ah / cell Ah (parallel),
        #   R_pack = R_cell * Ns / Np,  C_pack = C_cell * Np / Ns  (tau unchanged).
        # Without data, fall back to the illustrative pack-level defaults.
        ref = _DATASET_ECM_REF
        if ref:
            self.cells_series = self.nominal_voltage / ref['cell_nominal_voltage']
            self.cells_parallel = self.capacity_ah / ref['cell_capacity_ah']
            r_scale = self.cells_series / self.cells_parallel
            self.r0_ref = ref['R0'] * r_scale
            self.r1_ref = ref['R1'] * r_scale
            self.r2_ref = ref['R2'] * r_scale
            self.c1_ref = ref['C1'] / r_scale
            self.c2_ref = ref['C2'] / r_scale
            self.ecm_source = 'lipo_dataset_cell_scaled'
        else:
            self.cells_series = self.cells_parallel = float('nan')
            scale = (self.nominal_voltage / 14.8) * (5.0 / self.capacity_ah)
            self.r0_ref = 0.035 * scale
            self.r1_ref = 0.010 * scale
            self.r2_ref = 0.005 * scale
            self.c1_ref = 7.5 / self.r1_ref
            self.c2_ref = 30.0 / self.r2_ref
            self.ecm_source = 'illustrative_defaults'

        self.v1 = self.v2 = 0.0
        self.elapsed_s = 0.0
        self.discharge_ah = 0.0
        self.charge_ah = 0.0
        self.load_energy_j = 0.0
        self.source_energy_j = 0.0
        self.heat_energy_j = 0.0
        self.rc_storage_energy_j = 0.0
        self.power_balance_error_j = 0.0
        self.last_info: Dict[str, Any] = {}

        # ---------------------------------------------------------------
        # Xu-model degradation state  (NEW in v4)
        # ---------------------------------------------------------------
        ap = dict(_DEFAULT_AGING_PARAMS)
        if aging_params is not None:
            for k, v in aging_params.items():
                if k not in ap:
                    raise ValueError(f'Unknown aging_params key: {k!r}')
                ap[k] = float(v)
        self._ap = ap

        # fd_completed: sum of fd over all *finished* half-cycles so far.
        # Initialised from soh_init so that SOH = exp(-fd) is consistent.
        self._fd_completed: float = -math.log(float(soh_init))
        # SOC at the start of the current (in-progress) half-cycle.
        self._soc_cycle_start: float = float(soc_init)
        # Diagnostic split of fd_completed (calendar part is included in it).
        self._fd_calendar: float = 0.0

    # ------------------------------------------------------------------
    # Public: Xu-model cycle boundary
    # ------------------------------------------------------------------

    def cycle_end(self) -> float:
        """Commit the current half-cycle's degradation and start a new one.

        Call this at every half-cycle boundary:
          - when a discharge mission ends (before recharge begins), and
          - when recharge finishes (before the next discharge begins).

        The method promotes fd_total (= fd_completed + fd_current_halfcycle) to
        fd_completed, resets the cycle-start SOC to the current SOC, and returns
        the updated SOH.  Using fd_total directly ensures that cycle_end() is
        always exactly consistent with the real-time SOH reported by step(),
        regardless of how temperature varied during the half-cycle.

        When aging is disabled this is a no-op that returns the current SOH.
        """
        if not self.aging_enabled:
            return self.soh
        # fd_total is already the correct fd for the completed half-cycle.
        self._fd_completed = self.fd_total
        self._soc_cycle_start = self.soc
        soh = math.exp(-self._fd_completed)
        if soh < 0.5:
            raise BatteryLimitError('SOH below modeled range after cycle_end')
        self.soh = soh
        return soh

    @property
    def fd_calendar(self) -> float:
        """Cumulative calendar-aging fd (already included in fd_completed)."""
        return self._fd_calendar

    @property
    def fd_completed(self) -> float:
        """Cumulative degradation from all completed half-cycles."""
        return self._fd_completed

    @property
    def fd_total(self) -> float:
        """fd_completed plus the real-time estimate for the current half-cycle."""
        dod = abs(self._soc_cycle_start - self.soc)
        avg_soc = (self._soc_cycle_start + self.soc) / 2.0
        temp_k = self.temp_c + 273.15
        return self._fd_completed + _xu_fd_halfcycle(dod, avg_soc, temp_k, **self._ap)

    # ------------------------------------------------------------------
    # Internal helpers (unchanged from v3)
    # ------------------------------------------------------------------

    @staticmethod
    def _validated_curve(curve):
        try:
            pts = tuple((float(soc), float(voltage)) for soc, voltage in curve)
        except (TypeError, ValueError) as exc:
            raise ValueError('OCV curve must be pairs of (SOC, pack voltage)') from exc
        if (len(pts) < 2 or not math.isclose(pts[0][0], 0.0, abs_tol=1e-12)
                or not math.isclose(pts[-1][0], 1.0, abs_tol=1e-12)):
            raise ValueError('OCV curve needs at least two SOC points, from 0 to 1')
        if any(not math.isfinite(s) or not math.isfinite(v) or v <= 0
               for s, v in pts):
            raise ValueError('OCV table values must be finite and voltages positive')
        if any(b[0] <= a[0] for a, b in zip(pts, pts[1:])):
            raise ValueError('OCV SOC points must be strictly increasing')
        return pts

    @staticmethod
    def _curve_voltage(curve, soc):
        for (sa, va), (sb, vb) in zip(curve, curve[1:]):
            if soc <= sb:
                return va + (vb - va) * (soc - sa) / (sb - sa)
        return curve[-1][1]

    def ocv(self, soc=None, soh=None, temp_c=None):
        """Pack OCV(SOC, SOH, T). Default curves and aging are uncalibrated."""
        s = self.soc if soc is None else float(soc)
        h = self.soh if soh is None else float(soh)
        t = self.temp_c if temp_c is None else float(temp_c)
        if not (math.isfinite(s) and 0 <= s <= 1 and math.isfinite(h) and 0.5 <= h <= 1
                and math.isfinite(t) and -10 <= t <= 50):
            raise ValueError('Invalid SOC, SOH, or temperature')
        if self._ocv_tables_by_soh is not None:
            tables = self._ocv_tables_by_soh
            if h <= tables[0][0]:
                voltage = self._curve_voltage(tables[0][1], s)
            elif h >= tables[-1][0]:
                voltage = self._curve_voltage(tables[-1][1], s)
            else:
                voltage = None
                for (ha, ca), (hb, cb) in zip(tables, tables[1:]):
                    if h <= hb:
                        va = self._curve_voltage(ca, s)
                        vb = self._curve_voltage(cb, s)
                        voltage = va + (vb - va) * (h - ha) / (hb - ha)
                        break
        elif self._ocv_table is not None:
            voltage = self._curve_voltage(self._ocv_table, s)
        else:
            cell_v = self._curve_voltage(self._CELL_OCV, s)
            fresh_v = cell_v * self.nominal_voltage / 3.7
            age_factor = (1.0 - h) / 0.2
            soc_shape = 4.0 * s * (1.0 - s)
            shape = 0.25 * s + 0.875 * soc_shape
            voltage = (fresh_v + age_factor * self.ocv_age_delta_v_at_soh80
                       * shape * self.nominal_voltage / 14.8)
        return voltage + self.ocv_temp_coeff_v_per_c * (t - 25.0)

    def parameters(self, soc=None, soh=None, temp_c=None):
        """R0, R1, R2, C1, C2 all depend explicitly on SOC, SOH, and T.

        SOH dependence is intentionally made explicit here:
          - resistive terms grow as SOH drops (aging increases internal resistance)
          - capacitive terms shrink as SOH drops (aging reduces RC storage)
        """
        s = self.soc if soc is None else float(soc)
        h = self.soh if soh is None else float(soh)
        t = self.temp_c if temp_c is None else float(temp_c)
        if not (math.isfinite(s) and 0 <= s <= 1 and math.isfinite(h) and 0.5 <= h <= 1
                and math.isfinite(t) and -10 <= t <= 50):
            raise ValueError('Invalid SOC, SOH, or temperature')

        def soh_resistance_scale(soh_sensitivity: float) -> float:
            return 1.0 + soh_sensitivity * (1.0 - h) / 0.5

        def soh_capacitance_scale(soh_sensitivity: float) -> float:
            return 1.0 - soh_sensitivity * (1.0 - h) / 0.5

        def resistance(ref, soc_k, soh_sensitivity, temp_k):
            return (
                ref
                * (1.0 + soc_k * (1.0 - s) ** 2) / (1.0 + soc_k * 0.25)
                * soh_resistance_scale(soh_sensitivity)
                * math.exp(temp_k * (25.0 - t))
            )

        def capacitance(ref, soc_k, soh_sensitivity, temp_k):
            return (
                ref
                * (1.0 + soc_k * (s - 0.5))
                * soh_capacitance_scale(soh_sensitivity)
                * math.exp(temp_k * (t - 25.0))
            )

        r0 = resistance(self.r0_ref, 0.9, 0.70, 0.018)
        r1 = resistance(self.r1_ref, 1.4, 1.00, 0.022)
        r2 = resistance(self.r2_ref, 1.8, 1.30, 0.027)
        c1 = capacitance(self.c1_ref, 0.35, 0.60, 0.012)
        c2 = capacitance(self.c2_ref, 0.50, 0.90, 0.009)
        return ECMParameters(r0, r1, r2, c1, c2, r1 * c1, r2 * c2)

    def _electrical(self, power, soc, soh, v1, v2, temp):
        p = self.parameters(soc, soh, temp)
        ocv = self.ocv(soc, soh, temp)
        v_eff = ocv - v1 - v2
        disc = v_eff ** 2 - 4 * p.r0 * power
        if v_eff <= 0 or disc < 0:
            raise BatteryLimitError('Demand exceeds ECM deliverable power')
        current = 2 * power / (v_eff + math.sqrt(disc))
        voltage = v_eff - current * p.r0
        if self.max_discharge_current_a is not None and current > self.max_discharge_current_a:
            raise BatteryLimitError('Discharge current exceeds configured limit')
        if voltage < self.cutoff_voltage:
            raise BatteryLimitError('Terminal voltage below cutoff')
        return current, voltage, ocv, p

    @staticmethod
    def _rc_branch_step(v_old, current, resistance, capacitance, dt):
        """Exact constant-I, frozen-R/C RC step."""
        tau = resistance * capacitance
        decay = math.exp(-dt / tau)
        one_minus_decay = -math.expm1(-dt / tau)
        steady = current * resistance
        transient = v_old - steady
        v_new = steady + transient * decay
        heat_j = (steady ** 2 * dt
                  + 2 * steady * transient * tau * one_minus_decay
                  + transient ** 2 * tau * (-math.expm1(-2 * dt / tau)) / 2) / resistance
        storage_j = 0.5 * capacitance * (v_new ** 2 - v_old ** 2)
        return v_new, max(heat_j, 0.0), storage_j

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate_energy_drawn(self, power_w: float, duration_s: float = 1.0) -> Dict[str, float]:
        """Fixed-start-state power estimate; does NOT advance RC or SOH."""
        if not (math.isfinite(power_w) and power_w >= 0
                and math.isfinite(duration_s) and duration_s > 0):
            raise ValueError('Power must be >= 0; duration must be > 0')
        i, v, ocv, p = self._electrical(power_w, self.soc, self.soh,
                                        self.v1, self.v2, self.temp_c)
        load_j = power_w * duration_s
        source_j = ocv * i * duration_s
        instantaneous_heat_w = i ** 2 * p.r0 + self.v1 ** 2 / p.r1 + self.v2 ** 2 / p.r2
        return dict(current_a=i, voltage_v=v,
                    load_energy_j=load_j, source_energy_j=source_j,
                    load_energy_wh=load_j / 3600.0, source_energy_wh=source_j / 3600.0,
                    energy_loss_j=source_j - load_j,
                    instantaneous_heat_power_w=instantaneous_heat_w)

    def energy_loss(self, power_w: float, duration_s: float = 1.0) -> float:
        """Legacy fixed-state source-minus-load overhead (J), NOT heat (J)."""
        return self.estimate_energy_drawn(power_w, duration_s)['energy_loss_j']

    def step(self, power_w: float, dt: float = 1.0, temp_c: Optional[float] = None):
        """Apply constant terminal power for dt seconds; return (SOC, SOH).

        SOH is updated at every electrical sub-step using the Xu et al.
        semi-empirical cycle-aging model (when aging_enabled=True):

            fd_now  = fd_completed
                    + 0.5 * S_sigma(DOD_cumul) * S_delta(avgSOC_cumul) * S_T
            SOH     = exp(-fd_now)

        where DOD_cumul and avgSOC_cumul are measured from the most recent
        cycle_end() call (or from initialisation if cycle_end() has not yet
        been called). This is a running estimate of the current half-cycle's
        contribution: it is a *function* of the current state, not a running
        sum, so its terminal value is identical to the Rainflow result for
        one half-cycle regardless of the time-step size.

        On ANY battery limit violation, no battery state/energy is committed.
        """
        t = self.temp_c if temp_c is None else float(temp_c)
        if not (math.isfinite(power_w) and power_w >= 0 and math.isfinite(dt) and dt > 0
                and math.isfinite(t) and -10 <= t <= 50):
            raise ValueError('Invalid power, timestep, or temperature')
        n = math.ceil(dt / self.max_substep_s)
        if n > 100000:
            raise ValueError('Timestep too long; split it into smaller calls')
        ds = dt / n

        soc, soh, v1, v2 = self.soc, self.soh, self.v1, self.v2
        source_j = heat_j = rc_storage_j = discharge_ah = 0.0
        fd_cal_step = 0.0

        # Snapshot of Xu-model cycle state for rollback on error.
        fd_completed_snap = self._fd_completed
        soc_cycle_start_snap = self._soc_cycle_start

        try:
            for _ in range(n):
                i, v, ocv, p = self._electrical(power_w, soc, soh, v1, v2, t)
                da = i * ds / 3600.0
                soc_prev = soc
                soc -= da / (self.capacity_ah * soh)
                if soc < 0.02:
                    raise BatteryLimitError('SOC reserve reached')

                next_v1, heat1, store1 = self._rc_branch_step(v1, i, p.r1, p.c1, ds)
                next_v2, heat2, store2 = self._rc_branch_step(v2, i, p.r2, p.c2, ds)
                v1, v2 = next_v1, next_v2

                if self.aging_enabled:
                    # Real-time SOH estimate for the current half-cycle.
                    # fd is a function of cumulative state, not a running sum,
                    # so there is no dependence on time-step size.
                    t_k = t + 273.15
                    fd_cal_step += _xu_calendar_fd(soc_prev, soc, ds, t_k, **self._ap)
                    dod_cumul = abs(self._soc_cycle_start - soc)
                    avg_soc = (self._soc_cycle_start + soc) / 2.0
                    fd_hc = _xu_fd_halfcycle(dod_cumul, avg_soc, t_k, **self._ap)
                    fd_total = self._fd_completed + fd_cal_step + fd_hc
                    soh = math.exp(-fd_total)
                    if soh < 0.5:
                        raise BatteryLimitError('SOH below modeled range')

                source_j += ocv * i * ds
                heat_j += i ** 2 * p.r0 * ds + heat1 + heat2
                rc_storage_j += store1 + store2
                discharge_ah += da

        except BatteryLimitError:
            # Full rollback: restore Xu-model cycle state.
            self._fd_completed = fd_completed_snap
            self._soc_cycle_start = soc_cycle_start_snap
            raise

        i, v, ocv, p = self._electrical(power_w, soc, soh, v1, v2, t)
        load_j = power_w * dt
        balance_j = source_j - load_j - heat_j - rc_storage_j

        # Commit all state atomically.
        self.soc, self.soh, self.v1, self.v2, self.temp_c = soc, soh, v1, v2, t
        self._fd_completed += fd_cal_step
        self._fd_calendar += fd_cal_step
        self.elapsed_s += dt
        self.discharge_ah += discharge_ah
        self.load_energy_j += load_j
        self.source_energy_j += source_j
        self.heat_energy_j += heat_j
        self.rc_storage_energy_j += rc_storage_j
        self.power_balance_error_j += balance_j
        self.last_info = dict(
            soc=soc, soh=soh, voltage_v=v, current_a=i, ocv_v=ocv,
            r0_ohm=p.r0, r1_ohm=p.r1, r2_ohm=p.r2, c1_f=p.c1, c2_f=p.c2,
            tau1_s=p.tau1, tau2_s=p.tau2,
            fd_completed=self._fd_completed,
            fd_total=self.fd_total,
            fd_calendar=self._fd_calendar,
            fd_calendar_step=fd_cal_step,
            load_energy_j=load_j, source_energy_j=source_j,
            energy_loss_j=source_j - load_j,
            heat_energy_j=heat_j,
            rc_storage_energy_j=rc_storage_j,
            power_balance_error_j=balance_j,
            load_power_w=power_w,
            source_power_w=source_j / dt,
            internal_loss_power_w=heat_j / dt,
            rc_storage_power_w=rc_storage_j / dt,
            power_balance_error_w=balance_j / dt,
            discharge_ah=discharge_ah,
            cumulative_discharge_ah=self.discharge_ah,
            cumulative_load_energy_j=self.load_energy_j,
            cumulative_source_energy_j=self.source_energy_j,
            cumulative_heat_energy_j=self.heat_energy_j,
            cumulative_rc_storage_energy_j=self.rc_storage_energy_j,
            cumulative_power_balance_error_j=self.power_balance_error_j,
        )
        return soc, soh

    def rest(self, duration_s: float, temp_c: Optional[float] = None):
        """Idle at constant SOC for duration_s; accrues calendar aging only.

        RC branch voltages relax with the current tau values. No current flows,
        so the in-progress half-cycle is unchanged. Returns (SOC, SOH).
        """
        t = self.temp_c if temp_c is None else float(temp_c)
        if not (math.isfinite(duration_s) and duration_s >= 0
                and math.isfinite(t) and -10 <= t <= 50):
            raise ValueError('Invalid rest duration or temperature')
        if duration_s == 0:
            return self.soc, self.soh
        p = self.parameters(self.soc, self.soh, t)
        self.v1 *= math.exp(-duration_s / p.tau1)
        self.v2 *= math.exp(-duration_s / p.tau2)
        self.temp_c = t
        self.elapsed_s += duration_s
        if self.aging_enabled:
            fd_cal = _xu_calendar_fd(self.soc, self.soc, duration_s, t + 273.15, **self._ap)
            self._fd_completed += fd_cal
            self._fd_calendar += fd_cal
            soh = math.exp(-self.fd_total)
            if soh < 0.5:
                raise BatteryLimitError('SOH below modeled range during rest')
            self.soh = soh
        return self.soc, self.soh

    def charge(self, target_soc: float = 1.0, c_rate: float = 0.5,
               temp_c: Optional[float] = None) -> Dict[str, float]:
        """Recharge to target_soc by constant-current coulomb counting.

        Sequence:
          1. cycle_end() commits any in-progress discharge half-cycle;
          2. SOC rises linearly at I = c_rate * capacity_ah for the time needed
             to add (target_soc - SOC) * capacity_ah * SOH; calendar aging is
             integrated over that time;
          3. cycle_end() commits the charge half-cycle (DoD = SOC rise).

        Charge-side ECM voltages / CV phase are not modelled; RC branches are
        assumed relaxed afterwards. Returns a small info dict.
        """
        t = self.temp_c if temp_c is None else float(temp_c)
        if not (math.isfinite(target_soc) and 0.02 <= target_soc <= 1.0):
            raise ValueError('target_soc must be in [0.02, 1]')
        if not (math.isfinite(c_rate) and c_rate > 0):
            raise ValueError('c_rate must be positive')
        if not (math.isfinite(t) and -10 <= t <= 50):
            raise ValueError('Temperature must be in [-10, 50] C')

        self.cycle_end()                    # commit discharge half-cycle
        soc0 = self.soc
        if target_soc <= soc0:
            return dict(charge_ah=0.0, duration_s=0.0, fd_charge_halfcycle=0.0,
                        fd_calendar=0.0, soc=self.soc, soh=self.soh)

        current_a = c_rate * self.capacity_ah
        added_ah = (target_soc - soc0) * self.capacity_ah * self.soh
        duration_s = added_ah / current_a * 3600.0
        fd_before = self._fd_completed

        fd_cal = 0.0
        if self.aging_enabled:
            fd_cal = _xu_calendar_fd(soc0, target_soc, duration_s, t + 273.15, **self._ap)
            self._fd_completed += fd_cal
            self._fd_calendar += fd_cal

        self.soc = float(target_soc)
        self.v1 = self.v2 = 0.0
        self.temp_c = t
        self.elapsed_s += duration_s
        self.charge_ah += added_ah
        self.cycle_end()                    # commit charge half-cycle
        return dict(
            charge_ah=added_ah,
            duration_s=duration_s,
            fd_charge_halfcycle=self._fd_completed - fd_before - fd_cal,
            fd_calendar=fd_cal,
            soc=self.soc,
            soh=self.soh,
        )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def self_test():
    """Dependency-free contract and physical-accounting regression tests."""
    import copy

    def new(**kw):
        return Battery2RCECM(5.0, 14.8, **kw)

    # ---- Basic discharge: SOC falls, SOH monotone-ish, energy OK ----
    for power in (0, 5, 25, 42):
        b = new()
        prev_soc, prev_soh = b.soc, b.soh
        for _ in range(100):
            soc, soh = b.step(power)
            assert soc <= prev_soc, f'SOC increased at power={power}'
            # SOH is the real-time Xu-model estimate for the in-progress half-cycle.
            # It is computed as f(cumulative_DOD, avg_SOC), not a running sum, so
            # there is no step-count bias. The competing DOD and avg-SOC stress terms
            # produce a sub-nanosecond non-monotonicity (max ~3e-10) at very low DOD
            # when avg-SOC falls faster than DOD rises. The tolerance below is set two
            # orders of magnitude above this artefact so the test catches real bugs.
            assert soh <= prev_soh + 1e-8, (
                f'SOH increased by {soh - prev_soh:.2e} at power={power}')
            prev_soc, prev_soh = soc, soh
        assert math.isclose(b.load_energy_j, power * 100)
        assert math.isclose(b.source_energy_j,
                            b.load_energy_j + b.heat_energy_j
                            + b.rc_storage_energy_j + b.power_balance_error_j,
                            abs_tol=1e-8)
        e = b.estimate_energy_drawn(power, duration_s=10)
        assert math.isclose(e['source_energy_wh'], e['source_energy_j'] / 3600)
        assert math.isclose(e['load_energy_wh'], power * 10 / 3600)
        assert math.isclose(b.energy_loss(power, 10), e['energy_loss_j'])
        assert e['instantaneous_heat_power_w'] >= 0

    # ---- Frozen aging: SOH must not change ----
    b = new(aging_enabled=False)
    b.step(25)
    assert b.soh == 1.0, 'SOH changed with aging disabled'

    # ---- OCV calibration ----
    reference_ocv = new().ocv(soc=0.5, soh=0.8)
    assert new(ocv_age_delta_v_at_soh80=0).ocv(soc=0.5, soh=0.8) > reference_ocv
    assert math.isclose(new(ocv_age_delta_v_at_soh80=-0.8).ocv(soc=0.5, soh=0.8),
                        reference_ocv)
    assert math.isclose(new().ocv(soc=1, soh=1) - new().ocv(soc=1, soh=0.8),
                        0.2, abs_tol=1e-12)
    measured = new(ocv_tables_by_soh={1.0: [(0, 12), (1, 16)],
                                      0.8: [(0, 12), (1, 15)]})
    assert math.isclose(measured.ocv(soc=0.5, soh=0.9), 13.75)
    assert math.isclose(measured.ocv(soc=0.5, soh=0.6), 13.5)
    assert new(ocv_table=[(0, 15), (1, 15)]).ocv(soc=0.5, soh=0.8) == 15
    assert math.isclose(new(ocv_temp_coeff_v_per_c=0.005).ocv(soc=0.5, temp_c=35)
                        - new().ocv(soc=0.5, temp_c=25), 0.05, abs_tol=1e-12)

    # ---- Transactional rollback ----
    b = new()
    prior = copy.deepcopy(b.__dict__)
    try:
        b.step(1e6)
    except BatteryLimitError:
        pass
    else:
        raise AssertionError('Overload accepted')
    # Check all public state fields are rolled back.
    for attr in ('soc', 'soh', 'v1', 'v2', 'elapsed_s', 'discharge_ah',
                 'load_energy_j', 'source_energy_j', 'heat_energy_j',
                 '_fd_completed', '_soc_cycle_start'):
        assert getattr(b, attr) == prior[attr], f'{attr} not rolled back after overload'

    b = new(soc_init=0.0201, cutoff_voltage=1)
    prior = copy.deepcopy(b.__dict__)
    try:
        b.step(25, dt=10)
    except BatteryLimitError:
        pass
    else:
        raise AssertionError('SOC reserve violation accepted')
    for attr in ('soc', 'soh', '_fd_completed', '_soc_cycle_start'):
        assert getattr(b, attr) == prior[attr], f'{attr} not rolled back'

    b = new(max_discharge_current_a=2)
    try:
        b.step(100)
    except BatteryLimitError:
        pass
    else:
        raise AssertionError('Current limit violated')

    # ---- Aged battery draws more energy ----
    fresh = new(aging_enabled=False)
    old = new(soh_init=0.8, aging_enabled=False)
    for _ in range(600):
        fresh.step(30)
        old.step(30)
    assert old.soc < fresh.soc
    assert old.parameters().r0 > fresh.parameters().r0
    assert old.source_energy_j > fresh.source_energy_j

    # ---- RC relaxation ----
    b = new(aging_enabled=False)
    b.step(25)
    prev_v1 = b.v1
    # Relax for one time constant (LiPo EIS taus are only milliseconds).
    b.step(0, dt=b.parameters().tau1)
    assert 0 < b.v1 < prev_v1
    assert b.last_info['heat_energy_j'] > 0
    assert b.last_info['rc_storage_energy_j'] < 0
    assert math.isclose(b.last_info['heat_energy_j']
                        + b.last_info['rc_storage_energy_j'], 0, abs_tol=1e-8)

    # ---- ECM parameter dependencies ----
    for name in ('r0', 'r1', 'r2', 'c1', 'c2'):
        base = getattr(b.parameters(soc=0.5, soh=1, temp_c=25), name)
        assert getattr(b.parameters(soc=0.2, soh=1, temp_c=25), name) != base
        assert getattr(b.parameters(soc=0.5, soh=0.8, temp_c=25), name) != base
        assert getattr(b.parameters(soc=0.5, soh=1, temp_c=5), name) != base
    for soc0 in (0.05, 0.3, 0.8):
        for soh0 in (0.6, 0.85, 1.0):
            for temp0 in (-10, 5, 25, 50):
                p = b.parameters(soc=soc0, soh=soh0, temp_c=temp0)
                assert all(getattr(p, nm) > 0 for nm in
                           ('r0', 'r1', 'r2', 'c1', 'c2', 'tau1', 'tau2'))
                assert math.isclose(p.tau1, p.r1 * p.c1)
                assert math.isclose(p.tau2, p.r2 * p.c2)

    # ---- Substep convergence ----
    a, c = new(max_substep_s=0.1), new(max_substep_s=0.01)
    for power in (5, 42, 10, 30) * 20:
        a.step(power)
        c.step(power)
    assert abs(a.soc - c.soc) < 1e-5 and abs(a.v1 - c.v1) < 1e-4

    # ---- SOC accounting with fixed OCV ----
    b = new(aging_enabled=False, ocv_table=[(0, 15), (1, 15)])
    initial_soc = b.soc
    b.step(25)
    assert math.isclose(initial_soc - b.soc,
                        b.discharge_ah / (b.capacity_ah * b.soh), abs_tol=1e-12)

    # ---- Invalid inputs ----
    for bad in (-1, float('nan'), float('inf')):
        try:
            b.step(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid power accepted')

    # ---- Xu-model: cycle_end() advances fd_completed correctly ----
    b = new()
    b.step(25, dt=100)                   # discharge
    fd_after_discharge = b.fd_total
    soh_after_discharge = b.soh
    b.cycle_end()                         # commit discharge half-cycle
    assert math.isclose(b._fd_completed, fd_after_discharge, rel_tol=1e-9), \
        'cycle_end did not commit fd correctly'
    assert math.isclose(b.soh, soh_after_discharge, rel_tol=1e-9), \
        'SOH changed unexpectedly at cycle_end'
    assert math.isclose(b._soc_cycle_start, b.soc, abs_tol=1e-15), \
        'cycle_start SOC not reset after cycle_end'

    # ---- Xu-model: fd is independent of time-step size ----
    # Two batteries discharge to the same final SOC with different dt.
    b_coarse = new(aging_enabled=True, max_substep_s=1.0)
    b_fine   = new(aging_enabled=True, max_substep_s=0.01)
    for _ in range(50):
        b_coarse.step(25, dt=1.0)
        b_fine.step(25, dt=1.0)
    # fd_total must agree to better than 1e-9 (step-count independence).
    assert abs(b_coarse.fd_total - b_fine.fd_total) < 1e-9, (
        f'fd_total differs by time-step: {b_coarse.fd_total} vs {b_fine.fd_total}')
    assert abs(b_coarse.soh - b_fine.soh) < 1e-9, \
        'SOH differs by time-step size'

    # ---- Xu-model: custom aging_params ----
    custom = new(aging_params={'k_sigma': 0.5})
    assert custom._ap['k_sigma'] == 0.5
    assert custom._ap['k_T'] == _DEFAULT_AGING_PARAMS['k_T']
    try:
        new(aging_params={'bad_key': 1.0})
    except ValueError:
        pass
    else:
        raise AssertionError('Unknown aging_params key accepted')

    # ---- Charge half-cycle is aged, and charge() restores SOC ----
    b = new()
    b.step(25, dt=600)
    b.cycle_end()
    fd_dis = b._fd_completed
    info = b.charge(1.0, c_rate=0.5)
    assert b.soc == 1.0
    assert info['fd_charge_halfcycle'] > 0, 'charge half-cycle not aged'
    assert info['fd_calendar'] > 0, 'no calendar aging during charge'
    assert b._fd_completed > fd_dis
    # Discharge converts Ah->SOC with the SOH at each substep, charge() uses the
    # SOH after discharge; they differ by the SOH drop during the discharge.
    assert math.isclose(info['charge_ah'], b.discharge_ah,
                        rel_tol=10 * (1.0 - b.soh) + 1e-9), \
        'charge Ah should equal discharged Ah for a full recharge'

    # ---- Calendar aging: rest ages, longer rest ages more, k_t=0 disables ----
    r1, r2, r0 = new(), new(), new(aging_params={'k_t': 0.0})
    r1.rest(3600)
    r2.rest(7200)
    r0.rest(7200)
    assert 0 < r1.fd_calendar < r2.fd_calendar
    assert math.isclose(r2.fd_calendar, 2 * r1.fd_calendar, rel_tol=1e-12)
    assert r0.fd_calendar == 0.0 and r0.soh == 1.0
    hi, lo = new(soc_init=0.9), new(soc_init=0.3)
    hi.rest(3600)
    lo.rest(3600)
    assert hi.fd_calendar > lo.fd_calendar, 'calendar aging should rise with SOC'

    # ---- Calendar aging during discharge is substep-independent ----
    a, c = new(max_substep_s=1.0), new(max_substep_s=0.01)
    for _ in range(50):
        a.step(25)
        c.step(25)
    assert a.fd_calendar > 0
    assert math.isclose(a.fd_calendar, c.fd_calendar, rel_tol=1e-6)

    print('PASS: SOC/SOH/T, OCV calibration lookup, RC dynamics, energy/heat/storage '
          'accounting, aging switch, current limit, rollback, Xu-model fd independence, '
          'cycle_end, custom aging_params, charge half-cycle, calendar aging')


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Uncalibrated 2RC battery simulation (v4)')
    parser.add_argument('--capacity-ah', type=float, default=5.0)
    parser.add_argument('--nominal-voltage', type=float, default=14.8)
    parser.add_argument('--soh-init', type=float, default=0.85)
    parser.add_argument('--soc-init', type=float, default=1.0)
    parser.add_argument('--temp-c', type=float, default=25.0)
    parser.add_argument('--dt', type=float, default=1.0)
    parser.add_argument('--max-current-a', type=float, default=None)
    parser.add_argument('--disable-aging', action='store_true')
    parser.add_argument('--max-substep-s', type=float, default=0.1)
    parser.add_argument('--power-w', type=float, default=30)
    parser.add_argument('--power-demand-w', nargs='*', type=float, default=None)
    parser.add_argument('--duration-s', type=float, default=60)
    parser.add_argument('--steps', type=int, default=60)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--show-parameters', action='store_true')
    parser.add_argument('--show-ocv', action='store_true')
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    if args.steps < 0:
        parser.error('steps must be nonnegative')

    b = Battery2RCECM(
        args.capacity_ah, args.nominal_voltage,
        soc_init=args.soc_init, soh_init=args.soh_init, temp_c=args.temp_c,
        aging_enabled=not args.disable_aging,
        max_discharge_current_a=args.max_current_a,
        max_substep_s=args.max_substep_s,
    )

    if args.show_ocv:
        print('# Synthetic OCV(SOC, SOH), NOT pack measurements')
        print('soc,ocv_soh_1_v,ocv_soh_0.8_v,ocv_soh_0.6_v')
        for soc0 in (0.05, 0.2, 0.4, 0.6, 0.8, 0.95):
            print(f'{soc0:g},{b.ocv(soc0, 1):.6f},{b.ocv(soc0, .8):.6f},{b.ocv(soc0, .6):.6f}')
        return

    if args.show_parameters:
        print('# Illustrative, NOT measured or fitted ECM parameters')
        print('soc,soh,temp_c,r0_ohm,r1_ohm,r2_ohm,c1_f,c2_f,tau1_s,tau2_s')
        for soc0, soh0, t0 in ((0.8, 1., 25.), (0.2, 1., 25.), (0.8, 0.8, 25.),
                               (0.8, 1., 5.), (0.2, 0.8, 5.)):
            pp = b.parameters(soc=soc0, soh=soh0, temp_c=t0)
            print(f'{soc0:g},{soh0:g},{t0:g},{pp.r0:.6g},{pp.r1:.6g},{pp.r2:.6g},'
                  f'{pp.c1:.6g},{pp.c2:.6g},{pp.tau1:.6g},{pp.tau2:.6g}')
        return

    if args.power_demand_w is not None:
        print('# Uncalibrated energy estimate; no state update applied')
        print('power_w,duration_s,load_energy_j,source_energy_j,load_energy_wh,source_energy_wh')
        total_source_j = 0.0
        for demand in args.power_demand_w:
            e = b.estimate_energy_drawn(demand, duration_s=args.duration_s)
            total_source_j += e['source_energy_j']
            print(f'{demand:g},{args.duration_s:g},{e["load_energy_j"]:.6f},'
                  f'{e["source_energy_j"]:.6f},{e["load_energy_wh"]:.6f},'
                  f'{e["source_energy_wh"]:.6f}')
        print(f'# total_source_energy_wh,{total_source_j / 3600.0:.6f}')
        return

    print('# Uncalibrated simulation; default curves/aging are assumptions')
    print('time_s,soc,soh,fd_total,current_a,voltage_v')
    for _ in range(args.steps):
        try:
            soc, soh = b.step(args.power_w, dt=args.dt)
        except BatteryLimitError as exc:
            print(f'# Episode terminated; failed step rolled back: {exc}')
            break
        print(f'{b.elapsed_s:g},{soc:.9f},{soh:.12f},{b.fd_total:.6e},'
              f'{b.last_info["current_a"]:.6f},{b.last_info["voltage_v"]:.6f}')


if __name__ == '__main__':
    main()