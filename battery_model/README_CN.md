# Battery SOC/SOH 速度实验：纠正版本

## 修正
1. `battery_model.py`: Xu (2016) 的 `S_sigma` 输入应为平均 SOC；`S_delta` 输入应为 DOD。原来写反。
2. `parameter_fitting.py`: 同步修正上述公式；修复不同容量曲线的 cycle-index 未对齐问题，并避免 LiPO_3 的容量数据混入训练集。
3. `parameter_fitting.py`: 默认训练 LiPO_1、LiPO_2，验证 LiPO_3（根据现有数据集）。
4. `test_speed_mission.py`: `final_soc` 改成执行完最后一次任务、在充电之前的 SOC；保留充满后的 SOC 字段 `soc_after_recharge`。将默认 `max_missions` 增加至 100000（注意计算量很大）。
5. `diagnose_speed_aging.py`: 新增一次任务的快速诊断。
6. `test_stress_factors.py`: 新增退化公式的单调性和 SOC 记录回归测试。
7. `battery_model.py`: 新增日历老化 `d fd_cal/dt = k_t · S_sigma(SOC) · S_T`，在放电（`step`）、充电（`charge`）和静置（`rest`）期间连续累计；低速任务时间更长，因此会多付出日历老化。`k_t` 默认取 Xu 文献值 4.14e-10 1/s（LiPo 数据无法单独识别，属于假设，可通过 `aging_params={'k_t': ...}` 覆盖，设为 0 即关闭）。
8. `battery_model.py`: 新增 `charge(target_soc, c_rate)`：先提交放电半周期，再按恒流库仑计数充电、累计充电时长内的日历老化，最后提交充电半周期（DoD 取 |ΔSOC|）。充电侧 ECM 电压/CV 段未建模。
9. `battery_model.py`: 新增循环老化倍率 `k_cyc`（`fd_halfcycle = 0.5 · k_cyc · S_sigma · S_delta · S_T`），由拟合脚本给出。
10. `parameter_fitting.py`（审计版）: ECM 的 Q1/Q2 是 CPE 系数，用 Hsu-Mansfeld 公式换算为等效电容 (F)；老化只拟合有效衰减率 `SOH = exp(-k_eff·N)`（默认各电芯等权）。再按训练电芯应力循环的实测中位 DoD（≈0.87）与循环时长（≈4600 s），在 Xu 文献应力形状下把 `k_eff` 换算为 `k_cyc`（当前 ≈12.0），写入 `lipo_fit_parameters.json` 的 `degradation`，来源记录在 `degradation_provenance`。
12. `battery_model.py`: ECM 参数改为单体值按电池组拓扑缩放：`Ns = 组电压/3.7`，`Np = 组容量/单体容量`，`R_pack = R_cell·Ns/Np`，`C_pack = C_cell·Np/Ns`。旧格式 JSON（C1/C2 实为 CPE 系数）不再读取；原始 CSV 回退路径同样做 CPE→C 换算。注意 EIS 拟合的 RC 时间常数只有约 0.6 ms / 8 ms（高频弧），对 0.1 s 步长基本等同纯电阻，不含秒级以上的扩散动态。
11. `test_speed_mission.py`: 改为调用 `battery.charge()`，新增 `--charge-c-rate`、`--rest-s`、`--max-substep-s`；输出首次任务结束 SOC、EOL 任务结束 SOC 及日历老化占比。

## 运行
```bash
python test_stress_factors.py
python battery_model.py --self-test
python diagnose_speed_aging.py
python test_speed_soc.py
# 警告：完整的 EOL 仿真可能非常耗时
python test_speed_mission.py --speeds 1,2,3,4,5 --eol-soh 0.8 --max-missions 100000
```

如果要拟合 LiPo 数据，请将六个原始 zip 各自解压到 `LiPo_dataset/` 下，例如：
`LiPo_dataset/LiPO_1/`、`LiPo_dataset/LiPO_1_fit/` 等，并运行：
```bash
python parameter_fitting.py --train-datasets 1,2 --val-datasets 3
```

## 重要限制
- Xu 原论文的 SEI 非线性项未实现，仍是 `SOH = exp(-fd)` 的简化版；实测容量衰减后期加速，该形式无法拟合。
- LiPo 数据所有电芯使用同一应力工况（CC-CV 至 4.2 V，3 A 放电至截止，室温、无温度记录），DoD 只在约 0.76–0.96 之间且随循环数共变，平均 SOC 由 DoD 决定。因此 DoD/SOC/温度敏感系数**无法**由该数据识别，仍为 Xu（LMO 电池）文献值；速度实验中小 DoD（~0.1–0.3）的老化是外推结果。
- 各电芯单独的 `k_eff` 差异约 4.4 倍（LiPO_1 1.0e-3，LiPO_2 2.3e-4 /cycle）；LiPO_3 留出集 RMSE 6.7 pp，第 450 循环 SOH 实测 0.694、预测 0.806（单电芯速率范围 0.633–0.902）。寿命预测应视为量级估计。
- `k_t` 未经 LiPo 日历老化数据标定。
- 目前 25°C 恒温；没有电流影响的温升模型，因此没有通过温度体现负载的热应力。
- 从原始 `battery_model` 可以看到，如果缺少 `lipo_fit_parameters.json` 或 `LiPo_dataset`，模型会用 Xu 文献系数及示例 ECM 参数，而不是真实电池标定值。
- 单调性检验是在固定其他应力条件下进行，不能取代电芯实验验证。
