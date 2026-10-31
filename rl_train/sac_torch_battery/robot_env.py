import numpy as np
import random
from typing import List, Dict, Any, Tuple
import rainflow

class SoH_Rainflow_count:
    def __init__(self, SoH_0, SoC_data: np.ndarray, T_ref: float = 25.0):
        """
        SoC_data: 1D array-like of SoC samples (0..1 or 0..100 depending on your convention)
        T_ref:    Reference temperature in °C. Also used as default cycle temperature.
        """
        self.soc_data = np.asarray(SoC_data, dtype=float)

        # Model parameters
        self.k_sigma  = 1.04
        self.k_T      = 6.93e-2
        self.k_delta1 = 1.40e5
        self.k_delta2 = -5.01e-1
        self.k_delta3 = -1.23e5

        self.sigma_ref = 0.5  # Reference m SoC
        self.T_ref     = float(T_ref)
        self.SoH_0     = SoH_0

    def add_soc(self, soc: float):
        """
        Add a new SoC value to the soc_data array.

        Args:
            soc (float): The new SoC value to add.
        """
        self.soc_data = np.append(self.soc_data, soc)

    def perform_rainflow(self) -> List[Dict[str, Any]]:
        counted_cycles = rainflow.extract_cycles(self.soc_data)
        cycles: List[Dict[str, Any]] = []
        for rng, mean, count, i_start, i_end in counted_cycles:
            cycles.append({
                "DoD": rng,                 # depth of discharge (range)
                "mSoC": mean,               # mean SoC for the cycle
                "temperature": self.T_ref,  # use ref temp unless you have a profile
                "type": count,              # 0.5 for half cycles, 1.0 for full
                "start": i_start,
            })
        return cycles

    def stress_coefficients(self, cycle: Dict[str, Any]) -> Tuple[float, float, float]:
        sigma_i = float(cycle["DoD"])
        mSoC_i  = float(cycle["mSoC"])
        T_i     = float(cycle["temperature"])

        # Avoid divide-by-zero if T_i is 0 or very small
        if T_i <= 0:
            T_i = 1e-6

        S_sigma_i = np.exp(self.k_sigma * (mSoC_i - self.sigma_ref))
        S_T_i     = np.exp(self.k_T * (T_i - self.T_ref) * (self.T_ref / T_i))
        S_delta_i = 1.0 / (self.k_delta1 * (sigma_i ** self.k_delta2) * (self.T_ref / T_i) + self.k_delta3)

        return float(S_sigma_i), float(S_T_i), float(S_delta_i)

    def damage_function(self) -> float:
        cycles = self.perform_rainflow()
        f_d = 0.0

        for cycle in cycles:
            S_sigma_i, S_T_i, S_delta_i = self.stress_coefficients(cycle)
            u_i = float(cycle["type"])
            f_d += u_i * S_delta_i * S_sigma_i * S_T_i

        # New SoH (you can incorporate SoH_0 if you’re tracking progressive loss)
        SoH_new = float(np.exp(-f_d))
        return SoH_new

class RobotEnv:
    def __init__(self, soc=1.0):
        # cpu_fre_list = [499200, 652800, 806400, 960000, 1113600, 1267200, 1420800, 1574400, 1728000, 1881600, 2035200]
        cpu_fre_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # self.power_run_on_cpu = [4.48, 4.60, 4.81, 5.09, 5.27, 5.42, 5.61, 5.8, 5.99, 6.2, 6.5]
        self.power_run_on_cpu = [6.48, 6.60, 6.81, 7.09, 7.27, 7.42, 7.61, 7.8, 7.99, 8.2, 8.5]

        self.throughput = [19.86892096, 20.08365149, 21.29738784, 22.5030012, 23.72014441, 24.93324857, 25.14081996, 
                           26.35299931, 27.55472721, 28.75339697, 29.96143717]
        self.soc = soc
        self.R0 = 0.035
        self.R1 = 0.01 
        self.C1 = 2000.0
        self.R2 = 0.005
        self.C2 = 1500.0
        self.soh = 1.0
        self.V1 = 0.0
        self.V2 = 0.0
        self.Q = 5.0 * 3600.0 * self.soh  # Battery capacity in Ah
        self.dt = 1.0
        self.soh_cal = SoH_Rainflow_count(self.soh, self.soc, T_ref=25.0)

    def get_reward(self, thr, delta_soc, delta_soh, target_thr, speed):
        delta_soc= delta_soc * 100
        beta_1 = 550.0
        beta_2 = 1.0

        if thr >= target_thr or thr == 0:
            u = 1.0
        else:
            u = (thr / target_thr)**3.0

        soc_d = np.abs(delta_soc * beta_1) / speed
        soh_d = np.abs(delta_soh * beta_2) / speed
        print('get_reward()-speed = {}, soc_d = {}, soh_d = {}, u = {}'.format(
                            speed, soc_d, soh_d, u))
        # return u + beta / np.abs(delta_soc*1000/speed)
        return u - soc_d - soh_d

    def get_throughput(self, freq_index, speed):
        return self.throughput[freq_index] * (1 - 0.01 * speed)

    def get_sim_current_state(self, freq_index, speed):
        com_power = self.power_run_on_cpu[freq_index]
        mec_power = 1.14005 * speed + 0.108930 * speed * speed * speed + 1.1625
        thr = self.get_throughput(freq_index, speed)
        # self.update_battery_soc(mec_power, com_power)
        print('get_sim_current_state()-com_power = {}, mec_power = {}, thr = {}, soc = {}'.format(
                                        com_power, mec_power, thr, self.soc))
        return freq_index, speed, mec_power, com_power, thr, 25
    
    def getOCV(self): 
        return (
                -28.410 * pow(self.soc, 4)
                + 80.802 * pow(self.soc, 3)
                - 84.524 * pow(self.soc, 2)
                + 33.360 * self.soc
                + 9.963
            )

    def get_battery_delta_soc(self, mec_power, com_power):
        # Calculate the total power
        power_W = mec_power/0.8 + com_power
        ocv = self.getOCV()
        Rs = self.R0 / (self.soh*0.2)

        Vnet = ocv - self.V1 - self.V2
        discriminant = Vnet * Vnet - 4 * Rs * power_W

        if discriminant < 0:
            # Physically not possible — power too high for given voltage
            print("get_battery_soc()-Discriminant is negative, power too high for given voltage.")

        I = (Vnet - np.sqrt(discriminant)) / (2 * Rs)

        # update SOC
        delta_soc = -I * self.dt / self.Q
#         self.soc += delta_soc
        self.soh_cal.add_soc(self.soc+delta_soc)  # Add current SoC to the SoH calculator
        soh_prev = self.soh  # Preserve previous SoH for comparison
        self.soh = self.soh_cal.damage_function()
        delta_soh = np.exp(500*100*abs(self.soh - soh_prev)) # Calculate change in SoH
        self.Q = 5.0 * 3600.0  # Update battery capacity based on SoH
                
        # update v1, v2
        tau1 = self.R1 * self.C1
        self.V1 = self.V1 * np.exp(-self.dt/tau1) + I * self.R1 * (1 - np.exp(-self.dt/tau1))
        
        tau2 = self.R2 * self.C2
        self.V2 = self.V2 * np.exp(-self.dt/tau2) + I * self.R2 * (1 - np.exp(-self.dt/tau2))

        return delta_soc, delta_soh

    def update_battery_soc(self, mec_power, com_power):
        delta_soc, I = self.get_battery_delta_soc(mec_power, com_power)
#         self.soc += delta_soc
        
        # update v1, v2
        tau1 = self.R1 * self.C1
        self.V1 = self.V1 * np.exp(-self.dt/tau1) + I * self.R1 * (1 - np.exp(-self.dt/tau1))
        
        tau2 = self.R2 * self.C2
        self.V2 = self.V2 * np.exp(-self.dt/tau2) + I * self.R2 * (1 - np.exp(-self.dt/tau2))