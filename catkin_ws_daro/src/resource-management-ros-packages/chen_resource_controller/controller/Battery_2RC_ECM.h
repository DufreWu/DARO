#ifndef BATTERY_2RC_ECM_H
#define BATTERY_2RC_ECM_H

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <fstream>
#include <iomanip>

struct Cycle {
    double DoD;         // range
    double mSoC;        // mean SoC
    double temperature; // temperature used
    double type;        // 1.0 full, 0.5 half
    int start;          // indices in the input stream
    int end;
};

class StreamingSoH {
public:
    explicit StreamingSoH(double T_ref = 25.0)
        : T_ref_(T_ref) {}

    // Add one SoC sample (0..1 recommended). Temperature optional (defaults to T_ref_).
    // Returns updated SoH after processing any newly closed full cycles.
    double add_sample(double soc, double temperature = NAN) {
        if (std::isnan(temperature)) temperature = T_ref_;

        const int idx = static_cast<int>(count_samples_++);
        S_.emplace_back(idx, soc, temperature);

        // Try to close full cycles while the 4-point condition holds
        while (S_.size() >= 3) {
            const size_t n = S_.size();
            const auto& A = S_[n-3]; // (idx, soc, temp)
            const auto& B = S_[n-2];
            const auto& C = S_[n-1];

            const double R1 = std::fabs(B.soc - A.soc);
            const double R2 = std::fabs(C.soc - B.soc);

            if (R2 >= R1) {
                // Full cycle formed by A-B
                Cycle c;
                c.DoD         = R1;
                c.mSoC        = 0.5 * (A.soc + B.soc);
                c.temperature = choose_cycle_temp(A.temp, B.temp);
                c.type        = 1.0;
                c.start       = A.idx;
                c.end         = B.idx;

                accumulate_damage(c);

                // Remove middle point B (keep A and C)
                S_.erase(S_.end() - 2);
            } else {
                break;
            }
        }
        return get_soh();
    }

    // Count remaining half cycles (use at the end of a dataset or at a logical boundary).
    // Returns updated SoH.
    double finalize() {
        if (S_.size() >= 2) {
            for (size_t i = 0; i + 1 < S_.size(); ++i) {
                const auto& P = S_[i];
                const auto& Q = S_[i+1];
                Cycle c;
                c.DoD         = std::fabs(Q.soc - P.soc);
                c.mSoC        = 0.5 * (Q.soc + P.soc);
                c.temperature = choose_cycle_temp(P.temp, Q.temp);
                c.type        = 0.5;
                c.start       = P.idx;
                c.end         = Q.idx;
                accumulate_damage(c);
            }
        }
        S_.clear();
        return get_soh();
    }

    double get_soh() const {
        return std::exp(-f_d_);
    }

    double get_cumulative_damage() const { return f_d_; }

    // Optional: retrieve cycles that contributed to damage (for logging/analysis)
    const std::vector<Cycle>& cycles() const { return cycles_; }

    // ---- Parameter setters (match your Python constants) ----
    void set_params(double k_sigma, double k_T, double k_delta1, double k_delta2, double k_delta3,
                    double sigma_ref, double T_ref) {
        k_sigma_  = k_sigma;
        k_T_      = k_T;
        k_delta1_ = k_delta1;
        k_delta2_ = k_delta2;
        k_delta3_ = k_delta3;
        sigma_ref_ = sigma_ref;
        T_ref_     = T_ref;
    }

private:
    struct Node {
        int idx;
        double soc;
        double temp;
        Node(int i, double s, double t) : idx(i), soc(s), temp(t) {}
    };

    // Combine two temperatures for a cycle; you can change to min/max/instantaneous if preferred.
    static double choose_cycle_temp(double t1, double t2) {
        return 0.5 * (t1 + t2);
    }

    // Stress coefficients and damage accumulation
    void accumulate_damage(const Cycle& c) {
        double T_i = (c.temperature > 0.0) ? c.temperature : 1e-6; // avoid divide-by-zero
        const double S_sigma = std::exp(k_sigma_ * (c.mSoC - sigma_ref_));
        const double S_T     = std::exp(k_T_ * (T_i - T_ref_) * (T_ref_ / T_i));
        const double S_delta = 1.0 / (k_delta1_ * std::pow(c.DoD, k_delta2_) * (T_ref_ / T_i) + k_delta3_);

        f_d_ += c.type * S_delta * S_sigma * S_T;
        cycles_.push_back(c);
    }

private:
    // Rainflow stack (turning points). We push *every* sample; the ASTM logic
    // with R2 >= R1 will implicitly keep only reversals over time.
    std::vector<Node> S_;

    // Damage state
    double f_d_ = 0.0;
    size_t count_samples_ = 0;

    // Model parameters (same as your Python)
    double k_sigma_  = 1.04;
    double k_T_      = 6.93e-2;
    double k_delta1_ = 1.40e5;
    double k_delta2_ = -5.01e-1;
    double k_delta3_ = -1.23e5;

    double sigma_ref_ = 0.5;
    double T_ref_     = 25.0;

    // For inspection/logging
    std::vector<Cycle> cycles_;
};

class Battery2RCECM {
private:
    // battery parameters
    float R0;
    float R1;
    float C1;
    float R2;
    float C2;
    float Q;
    
    // battery state of charge parameters
    float soc;     // current soc (0-1)
    float dod;    // depth of discharge (0-1)
    float v1;      // RC1 voltage (V)
    float v2;      // RC2 voltage (V)
    float v_output; // output voltage (V)
    float temp;    // temperture (°C)
    int cycleCount; // circle count
    
    // battery state of health (SOH) parameters
    float capacityInitial; // initial capacity (Ah)
    float capacityNow;     // current capacity (Ah)
    float sohC;            // capacity health state (0-1)
    float sohR;            // internal resistance health state (0-1)
    
    // OCV-SOC mapping table
    std::vector<float> socTable = {1.00,  0.917, 0.833, 0.750, 0.667, 0.583, 0.500, 0.417, 0.333, 0.250, 0.167, 0.083};
    std::vector<float> ocvTable = {14.80, 14.77, 14.70, 14.61, 14.48, 14.33, 14.21, 14.06, 13.82, 13.48, 12.98, 11.88};

    StreamingSoH soh_class;

public:
    Battery2RCECM(float capacity_ah, float soc_init, float temp_c);

    float getOCV(float soc_value);
    float getSOC() const { return soc; }
    float getBatteryPower(float power_W);
    float calBatteryDeltaSOC(float power_W, float dt);
    float updateBatterySOH(float soc, float temp=25.0);
    float updateBatterySOC(float power_W, float dt);

    // print battery state
    void printState() const {
        std::cout << "Battery State:\n"
                  << "  SOC: " << soc*100 << "%\n"
                  << "  SOH_C: " << sohC*100 << "% (Capacity: " << capacityNow << "Ah)\n"
                  << "  SOH_R: " << sohR*100 << "% (Resistance multiplier: " << 1.0/sohR << "x)\n"
                  << "  Parameters: R0=" << R0*1000 << "mΩ, "
                  << "R1=" << R1*1000 << "mΩ, "
                  << "R2=" << R2*1000 << "mΩ\n"
                  << "  Cycles: " << cycleCount << ", Temperature: " << temp << "°C\n";
    }
};

#endif // BATTERY_2RC_ECM_H
