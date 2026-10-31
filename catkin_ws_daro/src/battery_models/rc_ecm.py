"""
Simple 2-RC equivalent circuit model (ECM) for batteries.
Provides a lightweight simulation API suitable for integration in tests and analysis.
"""

class TwoRCECM:
    """2-RC ECM.

    Params:
    - R0, R1, C1, R2, C2: resistances (Ohm) and capacitances (Farad)
    - capacity_ah: battery capacity in Ah
    """

    def __init__(self, R0=0.01, R1=0.02, C1=2000.0, R2=0.03, C2=5000.0, capacity_ah=2.5):
        self.R0 = float(R0)
        self.R1 = float(R1)
        self.C1 = float(C1)
        self.R2 = float(R2)
        self.C2 = float(C2)
        self.capacity_ah = float(capacity_ah)
        # charge in coulombs
        self.Q = self.capacity_ah * 3600.0

        # initial states
        self.v1 = 0.0
        self.v2 = 0.0
        self.soc = 1.0

    def ocv(self, soc: float) -> float:
        """Open-circuit voltage (OCV) approximation.

        This is a simple linear approximation; replace with a lookup or polynomial
        for higher fidelity.
        """
        soc = max(0.0, min(1.0, soc))
        return 3.0 + 1.2 * soc

    def step(self, I: float, dt: float) -> None:
        """Advance the model by dt seconds under current I (A).

        Convention: I > 0 means discharge (current out of the battery).
        """
        # time constants
        tau1 = max(1e-12, self.R1 * self.C1)
        tau2 = max(1e-12, self.R2 * self.C2)

        # steady-state voltages across each RC branch for constant current I
        v1_inf = self.R1 * I
        v2_inf = self.R2 * I

        # first-order exponential approach to steady-state
        self.v1 += ( -self.v1 + v1_inf ) * (dt / (tau1 + dt))
        self.v2 += ( -self.v2 + v2_inf ) * (dt / (tau2 + dt))

        # SOC update (I in A -> coulombs = A*s)
        self.soc -= (I * dt) / (self.Q + 1e-20)
        # clamp
        if self.soc < 0.0:
            self.soc = 0.0
        if self.soc > 1.0:
            self.soc = 1.0

    def voltage(self, I: float) -> float:
        """Compute terminal voltage under current I given internal states."""
        return self.ocv(self.soc) - self.R0 * I - self.v1 - self.v2

    def simulate(self, current_profile, dt: float):
        """Simulate a sequence of currents (iterable) with timestep dt (s).

        Returns list of terminal voltages (same length as current_profile).
        """
        voltages = []
        for I in current_profile:
            self.step(I, dt)
            voltages.append(self.voltage(I))
        return voltages


__all__ = ["TwoRCECM"]
