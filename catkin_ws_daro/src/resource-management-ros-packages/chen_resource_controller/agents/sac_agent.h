#ifndef SAC_AGENT_H
#define SAC_AGENT_H

#include <torch/torch.h>
#include <deque>
#include <random>
#include "boost/filesystem.hpp"
#include "boost/shared_ptr.hpp"
#include "sac_model.h"   // provides GaussianPolicy, TwinQ, hard_update, soft_update

namespace fs = boost::filesystem;

struct SACExperience {
    std::vector<float> state;
    std::vector<float> action;
    float reward;
    std::vector<float> next_state;
    bool done;
};

class SACAgent {
public:
    SACAgent(int state_size, int action_size,
             double lr_actor = 3e-4,
             double lr_critic = 3e-4,
             double lr_alpha = 3e-4,
             double gamma = 0.99,
             double tau = 5e-3)
    : state_size(state_size),
      action_size(action_size),
      policy(state_size, action_size),
      critic(state_size, action_size),
      target_critic(state_size, action_size),
      policy_optimizer(policy.parameters(), torch::optim::AdamOptions(lr_actor)),
      critic_optimizer(critic.parameters(), torch::optim::AdamOptions(lr_critic)),
      log_alpha(torch::tensor(0.0f, torch::requires_grad(true))),
      alpha_optimizer({log_alpha}, torch::optim::AdamOptions(lr_alpha)),
      discount_factor(gamma),
      tau(tau)
    {
        // Replay + misc hyperparams (kept close to your DDPG defaults)
        memory_maxlen = 500;
        batch_size = 64;
        train_start = 100;

        // Target entropy heuristic for tanh-Gaussian
        target_entropy = -static_cast<double>(action_size);

        // Initialize target critic = critic
        hard_update(target_critic, critic);
    }

    // Soft update after each gradient step
    void update_target_model_soft() {
        soft_update(target_critic, critic, tau);
    }

    void train_model() {
        if (memory.size() < static_cast<size_t>(train_start)) return;
        if (memory.size() < static_cast<size_t>(batch_size)) return;

        // ----- Sample batch
        std::vector<SACExperience> batch;
        std::sample(memory.begin(), memory.end(), std::back_inserter(batch),
                    batch_size, std::mt19937{std::random_device{}()});

        // Build tensors
        auto to_float = [](const std::vector<float>& v) {
            return torch::tensor(v, torch::TensorOptions().dtype(torch::kFloat32));
        };

        std::vector<torch::Tensor> s_list, ns_list, a_list;
        std::vector<float> r_list, d_list;

        s_list.reserve(batch_size);
        ns_list.reserve(batch_size);
        a_list.reserve(batch_size);
        r_list.reserve(batch_size);
        d_list.reserve(batch_size);

        for (const auto& e : batch) {
            s_list.push_back(to_float(e.state));
            ns_list.push_back(to_float(e.next_state));
            a_list.push_back(to_float(e.action));
            r_list.push_back(e.reward);
            d_list.push_back(e.done ? 1.0f : 0.0f);
        }

        auto states      = torch::stack(s_list);                      // [B, S]
        auto next_states = torch::stack(ns_list);                     // [B, S]
        auto actions     = torch::stack(a_list);                      // [B, A]
        auto rewards     = torch::tensor(r_list).unsqueeze(1);        // [B, 1]
        auto dones       = torch::tensor(d_list).unsqueeze(1);        // [B, 1]

        // Ensure float32
        states = states.to(torch::kFloat32);
        next_states = next_states.to(torch::kFloat32);
        actions = actions.to(torch::kFloat32);
        rewards = rewards.to(torch::kFloat32);
        dones = dones.to(torch::kFloat32);

        // ===== Critic update =====
        // Target: r + gamma * (1 - d) * ( min(Q1',Q2')(s',a') - alpha * logpi(a'|s') )
        {
            policy.forward(states);
            critic.forward(states, actions);

            auto sample_out = policy.sample(next_states); // (a', logp', mean_a')
            auto next_actions = std::get<0>(sample_out);
            auto logp_next    = std::get<1>(sample_out);

            auto q1_tq2 = target_critic.forward(next_states, next_actions);
            auto q1_t = q1_tq2.first;
            auto q2_t = q1_tq2.second;
            auto min_q_t = torch::min(q1_t, q2_t);

            auto alpha = torch::exp(log_alpha).detach();
            auto target = rewards + discount_factor * (1.0f - dones) * (min_q_t - alpha * logp_next);

            auto q1q2 = critic.forward(states, actions);
            auto q1 = q1q2.first;
            auto q2 = q1q2.second;

            auto loss_q1 = torch::mse_loss(q1, target.detach());
            auto loss_q2 = torch::mse_loss(q2, target.detach());
            auto critic_loss = loss_q1 + loss_q2;

            critic_optimizer.zero_grad();
            critic_loss.backward();
            torch::nn::utils::clip_grad_norm_(critic.parameters(), 1.0);
            critic_optimizer.step();
        }

        // ===== Policy update =====
        // Maximize E[min(Q1,Q2)(s, a_pi) - alpha * logpi]; equivalently minimize (alpha*logpi - minQ)
        torch::Tensor logp_pi, min_q_pi;
        {
            policy.forward(states);

            auto sample_out = policy.sample(states); // (a, logp, mean_a)
            auto a_pi   = std::get<0>(sample_out);
            logp_pi     = std::get<1>(sample_out);

            auto q1q2_pi = critic.forward(states, a_pi);
            min_q_pi = torch::min(q1q2_pi.first, q1q2_pi.second);

            auto alpha = torch::exp(log_alpha).detach();
            auto policy_loss = (alpha * logp_pi - min_q_pi).mean();

            policy_optimizer.zero_grad();
            policy_loss.backward();
            torch::nn::utils::clip_grad_norm_(policy.parameters(), 1.0);
            policy_optimizer.step();
        }

        // ===== Temperature (alpha) update =====
        {
            // J(alpha) = E[-alpha * (logpi + target_entropy)]
            auto alpha_loss = -(log_alpha * (logp_pi + target_entropy).detach()).mean();

            alpha_optimizer.zero_grad();
            alpha_loss.backward();
            alpha_optimizer.step();
        }

        // ===== Target soft update =====
        update_target_model_soft();
    }

    void step(std::vector<float> state, std::vector<float> action, float reward,
              std::vector<float> next_state, bool done)
    {
        SACExperience e{ std::move(state), std::move(action), reward, std::move(next_state), done };
        append_sample(e);

        if (memory.size() >= static_cast<size_t>(train_start)) {
            train_model();
        }
    }

    // Stochastic action during training; if eval_mode_ is true, return mean action.
    std::vector<float> get_action(const std::vector<float>& state) {
        std::vector<float> action(action_size);
        torch::Tensor state_tensor = torch::tensor(state, torch::dtype(torch::kFloat32)).unsqueeze(0); // [1, S]

        policy.eval();
        torch::NoGradGuard guard;
        auto out = policy.sample(state_tensor); // (a, logp, mean_a)
        torch::Tensor chosen = eval_mode_ ? std::get<2>(out) : std::get<0>(out);
        chosen = chosen.squeeze(0).to(torch::kCPU);

        for (int i = 0; i < action_size; ++i) action[i] = chosen[i].item<float>();
        return action;
    }

    void set_eval_mode(bool eval) { eval_mode_ = eval; }

    void append_sample(const SACExperience& e) {
        if (memory.size() >= static_cast<size_t>(memory_maxlen))
            memory.pop_front();
        memory.push_back(e);
    }

    // ==== Utilities ==============================================================
    void save_model(const std::string& directory) {
        if (!fs::exists(directory)) {
            fs::create_directories(directory);
        }

        // Save policy
        {
            torch::serialize::OutputArchive ar;
            policy.save(ar);
            std::string path = directory + "/sac_policy.pt";
            ar.save_to(path);
            std::cout << "Policy saved to: " << path << std::endl;
        }

        // Save critic (twin Q)
        {
            torch::serialize::OutputArchive ar;
            critic.save(ar);
            std::string path = directory + "/sac_critic.pt";
            ar.save_to(path);
            std::cout << "Critic saved to: " << path << std::endl;
        }

        // Save alpha
        {
            torch::serialize::OutputArchive ar;
            ar.write("log_alpha", log_alpha);
            std::string path = directory + "/sac_alpha.pt";
            ar.save_to(path);
            std::cout << "Alpha (log_alpha) saved to: " << path << std::endl;
        }
    }

    bool load_model(const std::string& directory) {
        try {
            // Load policy
            {
                torch::serialize::InputArchive ar;
                ar.load_from(directory + "/sac_policy.pt");
                policy.load(ar);
            }
            // Load critic
            {
                torch::serialize::InputArchive ar;
                ar.load_from(directory + "/sac_critic.pt");
                critic.load(ar);
            }
            // Load alpha
            {
                torch::serialize::InputArchive ar;
                ar.load_from(directory + "/sac_alpha.pt");
                ar.read("log_alpha", log_alpha);
            }
            // Reset target critic from critic
            hard_update(target_critic, critic);
            return true;
        } catch (const std::exception& ex) {
            std::cerr << "Failed to load SAC models: " << ex.what() << std::endl;
            return false;
        }
    }

    void hard_update(torch::nn::Module& target, const torch::nn::Module& source) {
        torch::NoGradGuard no_grad;
        auto tgt_params = target.named_parameters(true /*recurse*/);
        for (auto& kv : source.named_parameters(true)) {
            auto name = kv.key();
            auto it = tgt_params.find(name);
            if (it != nullptr) {
                it->copy_(kv.value());
            }
        }
        auto tgt_buffers = target.named_buffers(true);
        for (auto& kv : source.named_buffers(true)) {
            auto name = kv.key();
            auto it = tgt_buffers.find(name);
            if (it != nullptr) {
                it->copy_(kv.value());
            }
        }
    }

    void soft_update(torch::nn::Module& target, const torch::nn::Module& source, double tau) {
        torch::NoGradGuard no_grad;
        auto tgt_params = target.named_parameters(true);
        for (auto& kv : source.named_parameters(true)) {
            auto name = kv.key();
            auto it = tgt_params.find(name);
            if (it != nullptr) {
                it->mul_(1.0 - tau).add_(kv.value(), tau);
            }
        }
        auto tgt_buffers = target.named_buffers(true);
        for (auto& kv : source.named_buffers(true)) {
            auto name = kv.key();
            auto it = tgt_buffers.find(name);
            if (it != nullptr) {
                it->mul_(1.0 - tau).add_(kv.value(), tau);
            }
        }
    }

private:
    std::string model_dir;
    std::mt19937 rng;
    int state_size, action_size;

    // Replay buffer
    int memory_maxlen, batch_size, train_start;
    std::deque<SACExperience> memory;

    // Networks
    GaussianPolicy policy;
    TwinQ critic;
    TwinQ target_critic;

    // Optimizers
    torch::optim::Adam policy_optimizer;
    torch::optim::Adam critic_optimizer;

    // Entropy temperature (learned)
    torch::Tensor log_alpha; // requires_grad = true
    torch::optim::Adam alpha_optimizer;
    double target_entropy;

    // SAC hyperparams
    double discount_factor;
    double tau;

    // Action selection mode
    bool eval_mode_ = false;
};

#endif // PROJECT_SAC_AGENT_H
