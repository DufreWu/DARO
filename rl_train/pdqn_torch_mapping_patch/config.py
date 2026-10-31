# config.py for PDQN Robot Environment

robot_env = {
    'speed_min': 0.5,
    'speed_max': 5.0,

    'arm_freq_range': 11.0,
    'denver_freq_range': 11.0,
    'gpu_freq_range': 13.0,

    'cluster_num': 3,
    'app_num': 5,
    'target_thr': 25.0,
}

# 3^5 combinations
max_mapping_num = robot_env['cluster_num'] ** robot_env['app_num']  # 243

config = {
    # ---- STATE (must match RobotEnv.get_sim_current_state) ----
    'state_dim': (
        3                              # mech_power, comp_power, speed
        + robot_env['cluster_num']     # freq indices (3)
        + robot_env['app_num']         # mapping (5)
        + robot_env['app_num']         # throughput (5)
        + robot_env['app_num']         # qos (5)
    ),

    # ---- ACTION ----
    'action_dim': max_mapping_num,
    'action_parameter_dim': 2,  # [freq_control, speed_control]

    # ---- TRAINING ----
    'episodes': 3000,
    'steps_per_episode': 100,

    # ---- REWARD ----
    'beta': 5.0,

    # ---- STATE OFFSETS (for reward only) ----
    'state_thr_offset': 3 + robot_env['cluster_num'] + robot_env['app_num'],
    'state_qos_offset': 3 + robot_env['cluster_num'] + 2 * robot_env['app_num'],
}
