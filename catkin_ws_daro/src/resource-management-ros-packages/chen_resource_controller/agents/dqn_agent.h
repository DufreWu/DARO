#ifndef DQN_AGENT_H
#define DQN_AGENT_H

#include <torch/torch.h>
#include "boost/shared_ptr.hpp"
#include <vector>
#include <deque>
#include <tuple>
#include <random>
#include "dqn_model.h"

using DQNExperience = std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>;

class DQNAgent {
public:
    DQNAgent(int state_size, int action_size);
    int act(std::vector<float> state);
    void step(std::vector<float> state, int action, float reward, std::vector<float> next_state, bool done);
    void learn(std::vector<DQNExperience> experiences, float gamma);
    void save_model(const std::string& directory);
    bool load_model(const std::string& directory);
private:
    std::string model_dir;
    
    DQNModel::QNetwork qnetwork_local;
    DQNModel::QNetwork qnetwork_target;
    torch::optim::Adam optimizer;

    std::deque<DQNExperience> memory;
    int memory_size = 500;
    int batch_size = 64;
    
    float tau = 1e-3;
    float gamma = 0.99;
    int state_size, action_size;
    float epsilon, epsilon_min, epsilon_decay, discount_factor;
};

#endif // DQN_AGENT_H