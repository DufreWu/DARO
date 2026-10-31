#include "ppo_model.h"

ActorCritic::ActorCritic(int state_size, int action_size) {
    fc1 = register_module("fc1", torch::nn::Linear(state_size, 128));
    fc_logits = register_module("fc_logits", torch::nn::Linear(128, action_size));
    fc_value = register_module("fc_value", torch::nn::Linear(128, 1));
}

std::tuple<torch::Tensor, torch::Tensor> ActorCritic::forward(torch::Tensor x) {
    x = torch::relu(fc1->forward(x));
    auto logits = fc_logits->forward(x);
    auto value = fc_value->forward(x);
    return std::make_tuple(logits, value);
}
