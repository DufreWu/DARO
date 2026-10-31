#ifndef DQN_MODEL_H
#define DQN_MODEL_H

#include <torch/torch.h>

namespace DQNModel {

class QNetwork : public torch::nn::Module {
public:
    QNetwork(int state_size, int action_size) {
        fc1 = register_module("fc1", torch::nn::Linear(state_size, 128));
        fc2 = register_module("fc2", torch::nn::Linear(128, 64));
        fc3 = register_module("fc3", torch::nn::Linear(64, action_size));
    }
    torch::Tensor forward(torch::Tensor state) {
        state = torch::relu(fc1->forward(state));
        state = torch::relu(fc2->forward(state));
        return fc3->forward(state);
    }

private:
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};

}

#endif // DQN_MODEL_H
