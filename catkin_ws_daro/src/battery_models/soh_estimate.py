"""Streaming rainflow-based SOH estimator (Python port).

This module provides a `StreamingSoH` class that accepts SOC samples
and accumulates damage using the same formulas as the original C++
implementation. It exposes `add_sample`, `finalize`, `get_soh`, and
`get_cumulative_damage`.
"""

from dataclasses import dataclass
from math import exp, fabs
from typing import List


@dataclass
class Cycle:
    DoD: float
    mSoC: float
    temperature: float
    type_: float
    start: int
    end: int


@dataclass
class Node:
    idx: int
    soc: float
    temp: float


class StreamingSoH:
    def __init__(self, T_ref: float = 25.0):
        # Model parameters (defaults match the C++ values)
        # Fitted defaults (from fit_soh.py on LiPO_1)
        self.k_sigma = 2.1769839532016064
        self.k_T = 0.7329862358506106
        self.k_delta1 = 831.0027621097447
        self.k_delta2 = 0.5895853458185747
        self.k_delta3 = 63.11998499533556

        self.sigma_ref = 0.015345138626489023
        self.T_ref = float(T_ref)

        # Rainflow stack and state
        self.S: List[Node] = []
        self.count_samples = 0

        self.f_d = 0.0
        self.cycles: List[Cycle] = []

    def set_params(self, k_sigma, k_T, k_delta1, k_delta2, k_delta3, sigma_ref, T_ref):
        self.k_sigma = k_sigma
        self.k_T = k_T
        self.k_delta1 = k_delta1
        self.k_delta2 = k_delta2
        self.k_delta3 = k_delta3
        self.sigma_ref = sigma_ref
        self.T_ref = float(T_ref)

    @staticmethod
    def choose_cycle_temp(t1, t2):
        return 0.5 * (t1 + t2)

    def add_sample(self, soc: float, temperature: float = None) -> float:
        if temperature is None:
            temperature = self.T_ref

        idx = int(self.count_samples)
        self.count_samples += 1
        self.S.append(Node(idx, float(soc), float(temperature)))

        # Try to close full cycles while the 4-point condition holds
        while len(self.S) >= 3:
            A = self.S[-3]
            B = self.S[-2]
            C = self.S[-1]

            R1 = fabs(B.soc - A.soc)
            R2 = fabs(C.soc - B.soc)

            if R2 >= R1:
                c = Cycle(
                    DoD=R1,
                    mSoC=0.5 * (A.soc + B.soc),
                    temperature=self.choose_cycle_temp(A.temp, B.temp),
                    type_=1.0,
                    start=A.idx,
                    end=B.idx,
                )
                self.accumulate_damage(c)
                # remove middle point B
                del self.S[-2]
            else:
                break

        return self.get_soh()

    def finalize(self) -> float:
        # count remaining half cycles
        if len(self.S) >= 2:
            for i in range(len(self.S) - 1):
                P = self.S[i]
                Q = self.S[i + 1]
                c = Cycle(
                    DoD=fabs(Q.soc - P.soc),
                    mSoC=0.5 * (Q.soc + P.soc),
                    temperature=self.choose_cycle_temp(P.temp, Q.temp),
                    type_=0.5,
                    start=P.idx,
                    end=Q.idx,
                )
                self.accumulate_damage(c)

        self.S.clear()
        return self.get_soh()

    def get_soh(self) -> float:
        return exp(-self.f_d)

    def get_cumulative_damage(self) -> float:
        return self.f_d

    def accumulate_damage(self, c: Cycle) -> None:
        T_i = c.temperature if c.temperature is not None and c.temperature > 0.0 else 1e-6

        S_sigma = exp(self.k_sigma * (c.mSoC - self.sigma_ref))
        S_T = exp(self.k_T * (T_i - self.T_ref) * (self.T_ref / T_i))

        DoD_safe = max(c.DoD, 1e-12)
        S_delta = 1.0 / (
            self.k_delta1 * (DoD_safe ** self.k_delta2) * (self.T_ref / T_i) + self.k_delta3
        )

        self.f_d += c.type_ * S_delta * S_sigma * S_T
        self.cycles.append(c)


if __name__ == "__main__":
    # Simple demo: push some SOC turning points and finalize
    est = StreamingSoH()

    # emulate a few cycles of SOC samples (turning points)
    samples = [1.0, 0.8, 1.0, 0.6, 1.0]
    for s in samples:
        soh = est.add_sample(s)

    soh_final = est.finalize()
    print(f"Estimated SOH: {soh_final:.6f}")
    print(f"Cumulative damage: {est.get_cumulative_damage():.6e}")