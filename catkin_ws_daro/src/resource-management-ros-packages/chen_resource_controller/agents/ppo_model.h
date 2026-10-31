#ifndef PPO_MODEL_H
#define PPO_MODEL_H

#include <torch/torch.h>

class ActorCritic : public torch::nn::Module {
public:
    ActorCritic(int state_size, int action_size);

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x);
    std::pair<torch::Tensor, torch::Tensor> forward_actor(torch::Tensor state); // Add declaration
    torch::Tensor forward_critic(torch::Tensor state); // Add declaration

private:
    torch::nn::Linear fc1{nullptr}, fc_logits{nullptr}, fc_value{nullptr};
};

#endif // PPO_MODEL_H
