#ifndef PROJECT_SAC_MODEL_H
#define PROJECT_SAC_MODEL_H

#include <random>
#include <tuple>
#include <vector>
#include "torch/torch.h"

/**
 * SAC v2 modules (no value net):
 * - GaussianPolicy: outputs mean & log_std, supports reparameterized sampling with tanh squash.
 * - TwinQ: two independent Q networks (Q1, Q2).
 *
 * Notes:
 *  - Use the MIN of (Q1, Q2) for targets and policy loss.
 *  - Maintain target TwinQ for stable targets and update with Polyak averaging.
 *  - Optionally learn temperature alpha outside these modules via a learnable tensor.
 */

// ==== GaussianPolicy (Actor) =================================================
class GaussianPolicy : public torch::nn::Module {
public:
    GaussianPolicy(int64_t state_size,
                       int64_t action_size,
                       double log_std_min = -20.0,
                       double log_std_max = 2.0,
                       // optional scaling to env bounds; pass empty to keep [-1,1] from tanh
                       c10::optional<torch::Tensor> action_low = c10::nullopt,
                       c10::optional<torch::Tensor> action_high = c10::nullopt)
        : action_size_(action_size),
          LOG_STD_MIN_(log_std_min),
          LOG_STD_MAX_(log_std_max),
          action_low_(std::move(action_low)),
          action_high_(std::move(action_high)) {

        fc1 = register_module("fc1", torch::nn::Linear(state_size, 128));
        fc2 = register_module("fc2", torch::nn::Linear(128, 64));
        mean_head = register_module("mean_head", torch::nn::Linear(64, action_size));
        log_std_head = register_module("log_std_head", torch::nn::Linear(64, action_size));

        // Init last layers small to avoid saturation early
        torch::nn::init::uniform_(mean_head->weight, -1e-3, 1e-3);
        torch::nn::init::uniform_(mean_head->bias,   -1e-3, 1e-3);
        torch::nn::init::uniform_(log_std_head->weight, -1e-3, 1e-3);
        torch::nn::init::uniform_(log_std_head->bias,   -1e-3, 1e-3);
    }

    // returns (mean, log_std)
    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor state) {
        auto x = torch::relu(fc1->forward(state));
        x = torch::relu(fc2->forward(x));
        auto mean = mean_head->forward(x);
        auto log_std = log_std_head->forward(x);
        log_std = torch::clamp(log_std, LOG_STD_MIN_, LOG_STD_MAX_);
        return {mean, log_std};
    }

    // Reparameterized sampling with tanh squash.
    // Returns (action, log_prob, mean_action)
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> sample(torch::Tensor state, double eps = 1e-6) {
        auto [mean, log_std] = forward(state);
        auto std = torch::exp(log_std);

        // Reparameterization: u = mean + std * N(0, I)
        auto normal = torch::randn_like(mean);
        auto u = mean + std * normal;

        // Squash
        auto a = torch::tanh(u);

        // Log prob with tanh correction:
        // logπ(a|s) = logN(u; mean, std) - sum log(1 - tanh(u)^2 + eps)
        auto log_prob = (-0.5 * ((u - mean) / std).pow(2) - log_std - 0.5 * std::log(2 * M_PI)).sum(-1, true);
        auto correction = torch::log(1.0 - a.pow(2) + eps).sum(-1, true);
        log_prob = log_prob - correction;

        auto mean_action = torch::tanh(mean);

        if (action_low_.has_value() && action_high_.has_value()) {
            a = scale_to_bounds(a);
            mean_action = scale_to_bounds(mean_action);
        }

        return {a, log_prob, mean_action};
    }

private:
    torch::Tensor scale_to_bounds(const torch::Tensor& x) const {
        // x is in [-1,1]; scale to [low, high] elementwise.
        auto low = action_low_.value();
        auto high = action_high_.value();
        // Ensure broadcastable shapes: (1, action_size)
        auto l = low.dim() == 1 ? low.unsqueeze(0) : low;
        auto h = high.dim() == 1 ? high.unsqueeze(0) : high;
        return 0.5 * (x + 1.0) * (h - l) + l;
    }

    int64_t action_size_;
    double LOG_STD_MIN_, LOG_STD_MAX_;
    c10::optional<torch::Tensor> action_low_, action_high_;

public:
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, mean_head{nullptr}, log_std_head{nullptr};
};

// ==== Q Network(s) ===========================================================
class QNetworkImpl : public torch::nn::Module {
public:
    QNetworkImpl(int64_t state_size,
                 int64_t action_size,
                 int64_t hidden1 = 128,
                 int64_t hidden2 = 64) {
        fc1 = register_module("fc1", torch::nn::Linear(state_size + action_size, hidden1));
        fc2 = register_module("fc2", torch::nn::Linear(hidden1, hidden2));
        fc3 = register_module("fc3", torch::nn::Linear(hidden2, 1));
        torch::nn::init::uniform_(fc3->weight, -1e-3, 1e-3);
        torch::nn::init::uniform_(fc3->bias,   -1e-3, 1e-3);
    }

    torch::Tensor forward(torch::Tensor state, torch::Tensor action) {
        auto x = torch::cat({state, action}, -1);
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        return fc3->forward(x);
    }

public:
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};
TORCH_MODULE(QNetwork); // defines QNetwork as ModuleHolder<QNetworkImpl>

class TwinQ : public torch::nn::Module {
public:
    TwinQ(int64_t state_size, int64_t action_size) {
        q1 = register_module("q1", QNetwork(state_size, action_size, 128, 64));
        q2 = register_module("q2", QNetwork(state_size, action_size, 128, 64));
    }

    // (q1, q2)
    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor state, torch::Tensor action) {
        return { q1->forward(state, action), q2->forward(state, action) };
    }

public:
    QNetwork q1{nullptr}, q2{nullptr};
};

// ==== (Optional) Temperature parameter alpha ================================
// Many implementations keep alpha outside as an optimizer param.
// This is a tiny helper you can use if you want alpha to live in the model.
struct Alpha : torch::nn::Module {
    Alpha(double init_log_alpha = 0.0) {
        log_alpha = register_parameter("log_alpha", torch::tensor(init_log_alpha));
    }
    torch::Tensor log_alpha;
    torch::Tensor alpha() const { return torch::exp(log_alpha); }
};

#endif // PROJECT_SAC_MODEL_H
