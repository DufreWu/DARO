import numpy as np
from config import config, robot_env
from dataclasses import dataclass
from typing import List


# ============================================================
# Performance model
# ============================================================
@dataclass(frozen=True)
class PerfModel:
    thr_min: float
    thr_max: float
    thr_coeff: float
    pwr_coeff: float


# ============================================================
# Robot Environment
# ============================================================
class RobotEnv:
    def __init__(self):
        pass

    # ============================================================
    # FIXED mapping decode (independent of app_mask)
    # ============================================================
    def decode_mapping(self, mapping_index: int) -> List[int]:
        """
        Decode mapping_index into per-app cluster assignments.
        Fixed semantics: length = app_num always.
        """
        mapping = []
        for i in range(robot_env["app_num"]):
            cluster = (
                mapping_index //
                (robot_env["cluster_num"] ** i)
            ) % robot_env["cluster_num"]
            mapping.append(cluster)
        return mapping

    # ============================================================
    # Continuous → frequency indices
    # ============================================================
    def map_continuous_to_freq_indices(self, continuous_value):
        normalized = (continuous_value + 1.0) / 2.0

        total = (
            robot_env["arm_freq_range"]
            * robot_env["denver_freq_range"]
            * robot_env["gpu_freq_range"]
        )

        index = int(normalized * (total - 1))

        gpu = index % robot_env["gpu_freq_range"]
        temp = index // robot_env["gpu_freq_range"]
        denver = temp % robot_env["denver_freq_range"]
        arm = temp // robot_env["denver_freq_range"]

        return np.array([arm, denver, gpu], dtype=np.int32)

    # ============================================================
    # Reward
    # ============================================================
    def get_reward(self, state, beta, app_mask):
        mech_power = state[0] * 20.479
        comp_power = state[1] * 8.0
        speed = state[2] * (robot_env["speed_max"] - robot_env["speed_min"]) + robot_env["speed_min"]

        thr_list = state[
            config["state_thr_offset"]:
            config["state_thr_offset"] + robot_env["app_num"]
        ]
        qos_list = state[
            config["state_qos_offset"]:
            config["state_qos_offset"] + robot_env["app_num"]
        ]

        # Criticality
        ratios = []
        for i in range(robot_env["app_num"]):
            if app_mask[i] == 0:
                continue
            ratios.append(min(1.0, thr_list[i] / qos_list[i]))

        sys_crit = np.prod(ratios) if ratios else 1.0
        sys_crit = sys_crit ** (1.0 / len(ratios))

        # Energy per distance
        e_d = (mech_power + comp_power) / max(speed, 1e-3)
        energy_term = beta / e_d
        energy_term = min(energy_term, 2.0)
        reward = sys_crit + energy_term

        print(f"Mech Power: {mech_power:.2f}, Comp Power: {comp_power:.2f}, Speed: {speed:.2f}, E/D: {e_d:.2f}, Sys Crit: {sys_crit:.4f}")

        return reward, sys_crit, energy_term

    # ============================================================
    # Application performance
    # ============================================================
    @staticmethod
    def compute_app_perf(app_mask, mapping, freq_list, speed, max_freq, perf_table):
        num_apps = robot_env["app_num"]
        num_clusters = robot_env["cluster_num"]

        thr_list = [0.0] * num_apps
        total_power = 0.0

        # Normalize frequency by per-cluster maximum.
        # freq_norm ∈ (0, 1] represents relative DVFS intensity,
        # enabling unified modeling across heterogeneous clusters
        # (ARM / Denver / GPU) without explicit voltage modeling.
        freq_norm = np.array(freq_list) / np.array(max_freq)

        cluster_loads = np.zeros(num_clusters)
        for i in range(num_apps):
            if app_mask[i] == 1:
                cluster_loads[mapping[i]] += 1

        for i in range(num_apps):
            if app_mask[i] == 0:
                continue

            c = mapping[i]
            model = perf_table[i][c]
            load = max(cluster_loads[c], 1.0)

            # Application throughput model:
            # - Increases with normalized frequency
            # - Degrades under cluster-level contention (sub-linear penalty)
            # The exponent (<1) reflects partial overlap and cache effects
            thr = model.thr_coeff * freq_norm[c] / (load ** 0.9)
            thr = np.clip(thr, model.thr_min, model.thr_max)

            speed_scale = 1.0 - 0.3 * (speed - 0.5) / (5.0 - 0.5)
            speed_scale = np.clip(speed_scale, 0.8, 1.0)

            thr_list[i] = thr * speed_scale / 25.0

        prod = [1.0, 1.08, 1.25]
        for c in range(num_clusters):
            if cluster_loads[c] == 0:
                continue
            for i in range(num_apps):
                if app_mask[i] == 1 and mapping[i] == c:
                    # Computational power model:
                    # - Base power scaled by application- and cluster-specific coefficient
                    # - Amplified by cluster contention
                    # - Frequency-dependent term approximates a V^2 f relationship
                    #   where prod[c] implicitly captures cluster-level voltage
                    #   and architectural differences (e.g., GPU vs CPU cores)
                    total_power += (
                        (1.0 + cluster_loads[c] / 10.0)
                        * perf_table[i][c].pwr_coeff
                        * (freq_norm[c] * prod[c] * prod[c])
                    )

        # Add baseline SoC power and clip to realistic operating range
        # to prevent reward explosion and degenerate low-power policies
        total_power = float(np.clip(total_power + 4.0, 4.0, 8.0))

        return thr_list, total_power

    # ============================================================
    # Main transition
    # ============================================================
    def get_sim_current_state(self, action, action_params, app_mask):

        # ---- mapping (FIXED semantics) ----
        mapping = self.decode_mapping(action)

        # ---- frequency & speed ----
        freq_list = self.map_continuous_to_freq_indices(action_params[0])

        speed = (
            (action_params[1] + 1.0)
            * (robot_env["speed_max"] - robot_env["speed_min"]) / 2.0
            + robot_env["speed_min"]
        )

        mech_power = (
            1.14005 * speed
            + 0.10893 * speed ** 3
            + 1.1625
        )

        raw_thr, comp_power = self.compute_app_perf(
            app_mask,
            mapping,
            freq_list,
            speed,
            np.array([10.0, 10.0, 12.0]),
            self.perf_table,
        )

        qos_list = [
            robot_env["target_thr"] if app_mask[i] == 1 else 0.0
            for i in range(robot_env["app_num"])
        ]

        # ---- NORMALIZED STATE ----
        state = np.array([
            mech_power / 20.479,
            comp_power / 8.0,
            (speed - robot_env["speed_min"]) /
            (robot_env["speed_max"] - robot_env["speed_min"]),

            freq_list[0] / robot_env["arm_freq_range"],
            freq_list[1] / robot_env["denver_freq_range"],
            freq_list[2] / robot_env["gpu_freq_range"],

            * (np.array(mapping) / (robot_env["cluster_num"] - 1)),
            * raw_thr,
            * qos_list,
        ], dtype=np.float32)

        return state

    # ============================================================
    # Performance table
    # ============================================================
    perf_table = [
        [
            PerfModel(30, 37, 0.7, 0.9),
            PerfModel(31, 36, 0.9, 1.0),
            PerfModel(30, 35, 1.2, 1.4),
        ],
        [
            PerfModel(20, 29, 0.6, 0.7),
            PerfModel(22, 26, 0.8, 0.9),
            PerfModel(18, 26, 1.3, 1.3),
        ],
        [
            PerfModel(10, 29, 0.5, 0.6),
            PerfModel(12, 30, 0.7, 0.8),
            PerfModel(20, 29, 1.2, 1.4),
        ],
        [
            PerfModel(4, 19, 0.4, 0.6),
            PerfModel(5, 19, 0.6, 0.9),
            PerfModel(15, 28, 1.1, 1.5),
        ],
        [
            PerfModel(23, 29, 0.7, 0.8),
            PerfModel(25, 30, 0.9, 1.0),
            PerfModel(18, 29, 1.4, 1.6),
        ],
    ]
