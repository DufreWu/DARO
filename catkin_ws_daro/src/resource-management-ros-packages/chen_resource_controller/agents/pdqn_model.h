#ifndef PDQN_MODEL_H
#define PDQN_MODEL_H

#include <torch/torch.h>
#include <vector>
#include <random>
#include <tuple>

class ParamActor : public torch::nn::Module {
public:
    ParamActor(int state_dim, int param_dim, int hidden1_dim=128, int hidden2_dim=128) {
        // Define layers
        fc1 = register_module("fc1", torch::nn::Linear(state_dim, hidden1_dim));
        fc2 = register_module("fc2", torch::nn::Linear(hidden1_dim, hidden2_dim));
        fc3 = register_module("fc3", torch::nn::Linear(hidden2_dim, param_dim));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        x = torch::tanh(fc3->forward(x));  // Output bounded in [-1, 1]
        return x;
    }

private:
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};

class Q_CriticNetwork : public torch::nn::Module {
public:
    Q_CriticNetwork(int state_dim, int action_dim, int param_dim, int hidden1_dim=128, int hidden2_dim=128) {
        int input_dim = state_dim + param_dim;
        fc1 = register_module("fc1", torch::nn::Linear(input_dim, hidden1_dim));
        fc2 = register_module("fc2", torch::nn::Linear(hidden1_dim, hidden2_dim));
        fc3 = register_module("fc3", torch::nn::Linear(hidden2_dim, action_dim));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        return fc3->forward(x);
    }

private:
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};

#endif
