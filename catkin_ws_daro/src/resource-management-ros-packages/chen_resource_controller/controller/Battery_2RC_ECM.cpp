#include "Battery_2RC_ECM.h"


Battery2RCECM::Battery2RCECM(float capacity_ah = 5.0, float soc_init = 1.0, float temp_c = 25.0)
        : capacityInitial(capacity_ah), soc(soc_init), dod(0.0), temp(temp_c), cycleCount(0),
          R0(0.035), R1(0.01), C1(2000.0), R2(0.005), C2(1500.0),
          sohC(0.85), sohR(0.85), capacityNow(capacity_ah), v1(0.0), v2(0.0) {

    Q = capacityNow * 3600.0 * sohC;
    soh_class = StreamingSoH(25.0);
    printState();
}

float Battery2RCECM::getOCV(float soc_value) {
    if (soc_value < 0) soc_value = soc;
    
    return (
            -28.410 * pow(soc, 4)
            + 80.802 * pow(soc, 3)
            - 84.524 * pow(soc, 2)
            + 33.360 * soc
            + 9.963
        ) / sohC;
}

float Battery2RCECM::getBatteryPower(float power_W) {
    float voc = getOCV(soc);
    float Rs = R0 / (sohR*0.2);  // degradation-aware, TODO: consider battery age health

    float Vnet = voc - v1 - v2;
    float discriminant = Vnet * Vnet - 4 * Rs * power_W;

    if (discriminant < 0) {
        // Physically not possible — power too high for given voltage
        throw std::runtime_error("Power too high: invalid discriminant in current calculation");
    }

    float I = (Vnet - std::sqrt(discriminant)) / (2 * Rs);
    float V_output = voc - I * Rs - v1 - v2;
    float P_batt = voc * I;

    return P_batt;
}

float Battery2RCECM::calBatteryDeltaSOC(float power_W, float dt) {
    float voc = getOCV(soc);
    float Rs = R0 / (sohR*0.2); 

    float Vnet = voc - v1 - v2;
    float discriminant = Vnet * Vnet - 4 * Rs * power_W;

    if (discriminant < 0) {
        // Physically not possible — power too high for given voltage
        throw std::runtime_error("Power too high: invalid discriminant in current calculation");
    }

    float I = (Vnet - std::sqrt(discriminant)) / (2 * Rs);
    float delta_soc = -I * dt / Q;

    return delta_soc;
}

float Battery2RCECM::updateBatterySOH(float soc, float temp) {
    float current_soh = soh_class.add_sample(soc, temp);

    return this->sohC - (1.0 - current_soh);
}

float Battery2RCECM::updateBatterySOC(float power_W, float dt) {
    float voc = getOCV(soc);
    float Rs = R0 / (sohR*0.2);  // degradation-aware, TODO: consider battery age health

    float Vnet = voc - v1 - v2;
    power_W = power_W * 1.2;
    float discriminant = Vnet * Vnet - 4 * Rs * power_W;

    if (discriminant < 0) {
        // Physically not possible — power too high for given voltage
        throw std::runtime_error("Power too high: invalid discriminant in current calculation");
    }

    float I = (Vnet - std::sqrt(discriminant)) / (2 * Rs);
    // update SOC
    float delta_soc = -I * dt / Q;
    soc += delta_soc;
    dod += std::abs(delta_soc);
    
    // update v1, v2
    float tau1 = R1 * C1;
    v1 = v1 * std::exp(-dt/tau1) + I * R1 * (1 - std::exp(-dt/tau1));
    
    float tau2 = R2 * C2;
    v2 = v2 * std::exp(-dt/tau2) + I * R2 * (1 - std::exp(-dt/tau2));

    return soc;
}