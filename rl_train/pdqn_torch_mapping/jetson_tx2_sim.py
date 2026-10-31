from dataclasses import dataclass
from config import config, robot_env
import numpy as np

@dataclass(frozen=True)
class PerfModel:
    thr_min: float     # minimum throughput
    thr_max: float     # maximum throughput
    thr_coeff: float   # throughput scaling coefficient
    pwr_coeff: float   # power coefficient

perf_table = [
    # edge_image
    [
        PerfModel(20, 37, 0.5, 0.9),    # ARM
        PerfModel(21, 36, 0.8, 1.1),    # DENVER
        PerfModel(16, 35, 1.2, 1.4),    # GPU
    ],
    # denoising_filling
    [
        PerfModel(20, 29, 0.7, 0.9),    # ARM
        PerfModel(22, 26, 0.9, 1.2),    # DENVER
        PerfModel(18, 27, 1.3, 1.5),    # GPU
    ],
    # distance_surface
    [
        PerfModel(10, 29, 0.6, 1.0),    # ARM
        PerfModel(12, 30, 0.7, 1.2),    # DENVER
        PerfModel(20, 29, 1.2, 1.5),    # GPU
    ],
    # optical_flow
    [
        PerfModel(4, 19, 1.0, 1.2),     # ARM
        PerfModel(5, 19, 1.1, 1.3),     # DENVER
        PerfModel(15, 28, 1.2, 1.5),    # GPU
    ],
    # tracker
    [
        PerfModel(23, 29, 0.70, 0.85),    # ARM
        PerfModel(25, 30, 0.90, 1.05),    # DENVER
        PerfModel(18, 29, 1.40, 1.60),    # GPU
    ],
]

@staticmethod
def compute_app_perf(app_mask, mapping, freq_list, max_freq, perf_table):
    """
    Returns:
        thr_list    : length-5 throughput list (inactive → 0)
        total_power : scalar total system power
    """

    num_apps = robot_env['app_num']   # safer than hardcoding 5
    num_clusters = robot_env['cluster_num']

    thr_list = [0.0] * num_apps
    total_power = 0.0

    # --------------------------------------------------------
    # Frequency normalization
    # --------------------------------------------------------
    freq_norm = np.array(freq_list) / np.array(max_freq)

    # --------------------------------------------------------
    # Cluster loads (contention)
    # --------------------------------------------------------
    cluster_loads = np.zeros(num_clusters)
    for i in range(len(app_mask)):
        if app_mask[i] == 1 and mapping[i] != -1:
            cluster_loads[mapping[i]] += 1

    # --------------------------------------------------------
    # Per-application throughput
    # --------------------------------------------------------
    for i in range(num_apps):
        if app_mask[i] == 0:
            continue

        c = mapping[i]
        if c == -1:
            continue

        model = perf_table[i][c]
        load = max(cluster_loads[c], 1.0)  # safety

        # Throughput ↓ with load
        thr = model.thr_coeff * freq_norm[c] / load
        thr = min(max(thr, model.thr_min), model.thr_max)
        thr_list[i] = thr

    # --------------------------------------------------------
    # Power (system-level, cluster-based)
    # --------------------------------------------------------
    for c in range(num_clusters):
        if cluster_loads[c] == 0:
            continue

        # Power ↑ with load
        coeffs = [
            perf_table[i][c].pwr_coeff
            for i in range(num_apps)
            if app_mask[i] == 1 and mapping[i] == c
        ]

        if len(coeffs) == 0:
            continue

        total_power += (
            cluster_loads[c]
            * np.mean(coeffs)
            * (freq_norm[c] ** 2)
        )

    total_power += 4.0
    total_power = float(np.clip(total_power, 4.0, 9.0))

    return thr_list, total_power

app_mask  = [0, 0, 0, 0, 0]
mapping   = [0, 0, 0, 0, 0]

freq_list = [10.0, 10.0, 12.0]     # ARM, DENVER, GPU
max_freq  = [10.0, 10.0, 12.0]

thr, pwr = compute_app_perf(
    app_mask,
    mapping,
    freq_list,
    max_freq,
    perf_table
)

print("Throughput:", thr)
print("Total power:", pwr)

import numpy as np
import matplotlib.pyplot as plt

# Data extracted from your perf_table
apps = ["Edge Image", "Denoising", "Distance Surface", "Optical Flow", "Tracker"]
clusters = ["ARM", "Denver", "GPU"]

# Reshaping coefficients for easier plotting
# thr_coeff: [app_index][cluster_index]
thr_coeffs = np.array([
    [0.5, 0.8, 1.2],
    [0.7, 0.9, 1.3],
    [0.6, 0.7, 1.2],
    [1.0, 1.1, 1.2],
    [0.7, 0.9, 1.4]
])

def visualize_perf_table():
    freq_norm = np.linspace(0.1, 1.0, 100)
    loads = [1, 2, 4]  # Compare light vs heavy contention
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Throughput for a specific app (e.g., Optical Flow) across clusters
    app_idx = 3 
    for c_idx, cluster in enumerate(clusters):
        for load in loads:
            # Formula: thr = coeff * freq / load
            thr = (thr_coeffs[app_idx, c_idx] * freq_norm) / load
            axes[0, 0].plot(freq_norm, thr, label=f"{cluster} (Load {load})")
    
    axes[0, 0].set_title(f"Throughput: {apps[app_idx]}")
    axes[0, 0].set_ylabel("Throughput (Normalized)")
    axes[0, 0].legend(fontsize='small', ncol=2)

    # 2. Power Consumption (Quadratic Scaling)
    # Formula: load * mean_coeff * freq^2
    avg_pwr_coeff = 1.1 # Representative average
    for c_idx, cluster in enumerate(clusters):
        for load in loads:
            pwr = load * avg_pwr_coeff * (freq_norm ** 2)
            axes[0, 1].plot(freq_norm, pwr, label=f"{cluster} (Load {load})")
            
    axes[0, 1].set_title("Dynamic Power Consumption (Quadratic)")
    axes[0, 1].set_ylabel("Power Units")
    
    # 3. Efficiency (Throughput per Watt)
    # Formula: (coeff / load) / (load * coeff * freq) -> scales as 1 / (load^2 * freq)
    for c_idx in range(len(clusters)):
        eff = (thr_coeffs[app_idx, c_idx] * freq_norm) / (1 * avg_pwr_coeff * freq_norm**2)
        axes[1, 0].plot(freq_norm[10:], eff[10:], label=clusters[c_idx])
        
    axes[1, 0].set_title("Efficiency (Perf/Watt) vs Frequency")
    axes[1, 0].set_xlabel("Normalized Frequency")
    axes[1, 0].legend()

    # 4. App Comparison on GPU
    for i, app in enumerate(apps):
        thr = (thr_coeffs[i, 2] * freq_norm) # GPU load = 1
        axes[1, 1].plot(freq_norm, thr, label=app)
        
    axes[1, 1].set_title("All Apps on GPU (No Contention)")
    axes[1, 1].set_xlabel("Normalized Frequency")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()

visualize_perf_table()