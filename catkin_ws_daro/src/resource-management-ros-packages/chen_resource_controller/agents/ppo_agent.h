#ifndef PPO_AGENT_H
#define PPO_AGENT_H

#include "ppo_model.h"
#include <vector>
#include <memory>
#include <torch/torch.h>

struct Trajectory {
    torch::Tensor state;
    torch::Tensor action;
    torch::Tensor log_prob;
    torch::Tensor reward;
    torch::Tensor value;
    torch::Tensor done;
};

class PPOAgent {
public:
    PPOAgent(int state_size, int action_size);

    int act(std::vector<float> state);  // Select action based on current policy
    void store(const Trajectory& traj); // Store transition (optional use)
    void step(const std::vector<float>& next_state, float reward, bool done); // Update reward/done and trigger learn
    void learn(); // Run PPO update on stored trajectories
    void save_model(const std::string& directory);

private:
    std::shared_ptr<ActorCritic> model;
    torch::optim::Adam optimizer;

    int state_size;
    int action_size;
    float gamma;
    float clip_param;
    const size_t rollout_length = 128;  // Triggers learning every 128 steps
    std::vector<Trajectory> memory;     // Trajectory buffer
};

#endif // PPO_AGENT_H
