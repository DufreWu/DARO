#ifndef PDQN_AGENT_H
#define PDQN_AGENT_H

#include <torch/torch.h>
#include <boost/filesystem.hpp>
#include <vector>
#include <deque>
#include <tuple>
#include <random>
#include "pdqn_model.h"

namespace fs = boost::filesystem;

struct PDQNExperience {
    std::vector<float> state;
    int action;
    std::vector<float> action_params;
    float reward;
    std::vector<float> next_state;
    int done;
};

class PDQNAgent {
public:
    PDQNAgent(int state_size, int action_size, int action_parameter_size)
        : state_size(state_size), action_size(action_size),
          action_parameter_size(action_parameter_size),
          actor_model(state_size, action_parameter_size),
          target_actor_model(state_size, action_parameter_size),
          q_model(state_size, action_size, action_parameter_size),
          target_q_model(state_size, action_size, action_parameter_size),
          actor_optimizer(actor_model.parameters(), torch::optim::AdamOptions(0.0005)),
          q_optimizer(q_model.parameters(), torch::optim::AdamOptions(0.002)) {

        epsilon = 1.0f;
        epsilon_min = 0.05f;
        epsilon_decay = 0.95f;
        discount_factor = 0.99f;
        memory_maxlen = 1000;
        batch_size = 64;
        train_start = 100;
        tau_q = 0.99;
        tau_actor = 0.99;

        std::cout << "[PDQNAgent] Initialized with state_size=" << state_size
                  << ", action_size=" << action_size
                  << ", action_parameter_size=" << action_parameter_size << std::endl;
    }

    void update_target_model() {
        // Copy actor weights
        {
            torch::NoGradGuard no_grad;
            auto actor_params = actor_model.named_parameters();
            auto target_actor_params = target_actor_model.named_parameters(true);

            for (const auto& item : actor_params) {
                const auto& name = item.key();
                if (target_actor_params.contains(name)) {
                    target_actor_params[name].copy_(
                        tau_actor * item.value() + (1.0 - tau_actor) * target_actor_params[name]
                    );
                }
            }
        }

        // Copy critic weights
        {
            torch::NoGradGuard no_grad;
            auto q_params = q_model.named_parameters();
            auto target_q_params = target_q_model.named_parameters(true);

            for (const auto& item : q_params) {
                const auto& name = item.key();
                if (target_q_params.contains(name)) {
                    target_q_params[name].copy_(
                        tau_q * item.value() + (1.0 - tau_q) * target_q_params[name]
                    );
                }
            }
        }
    }

    std::pair<int, std::vector<float>> get_action(const std::vector<float>& state, bool is_training) {
        torch::Tensor state_tensor = torch::tensor(state, torch::dtype(torch::kFloat32));
        torch::Tensor action_param_tensor = actor_model.forward(state_tensor).detach();
        std::vector<float> action_params(
            action_param_tensor.data_ptr<float>(),
            action_param_tensor.data_ptr<float>() + action_param_tensor.numel()
        );
        int action = 0;
        std::string log_mode;

        if (is_training == true) {
            std::uniform_real_distribution<float> real_dist(0.0f, 1.0f);
            float epsilon_threshold = real_dist(rng);  // Random float in [0,1]
            std::cout << "[Actor] state_tensor: " << state_tensor 
                    << ", action_param: " << action_params 
                    << ", epsilon_threshold: " << epsilon_threshold
                    << ", epsilon: " << epsilon << std::endl;
            if (epsilon_threshold <= epsilon) {
                std::uniform_int_distribution<> int_dist(0, action_size - 1);
                action = int_dist(rng);
                std::cout << "[Training Exploitation] action = " << action;

                // Add uniform + Gaussian noise to each parameter
                std::uniform_real_distribution<float> uniform_noise(-1.0f, 1.0f);
                std::normal_distribution<float> gaussian_noise(0.0f, 0.1f);

                for (auto& p : action_params) {
                    p = std::clamp(uniform_noise(rng) + gaussian_noise(rng), -1.0f, 1.0f);
                }
            } else {
                torch::Tensor q_input = torch::cat({state_tensor, action_param_tensor}, 0).unsqueeze(0);  // [1, state+1]
                torch::Tensor q_values = q_model.forward(q_input);

                // std::cout << "[Q] q_input: " << q_input
                //       << ", q_values: " << q_values << std::endl;

                action = q_values.argmax(1).item<int>();
            }
            log_mode = "Exploration";
        } else {
            torch::Tensor q_input = torch::cat({state_tensor, action_param_tensor}, 0).unsqueeze(0);  // [1, state+1]
            torch::Tensor q_values = q_model.forward(q_input);
            action = q_values.argmax(1).item<int>();
            log_mode = "Exploitation";
        }

        // ===== Logging =====
        std::cout << "[" << (is_training ? "Training " : "Inference ")
                << log_mode << "] action=" << action << ", params=[";
        for (size_t i = 0; i < action_params.size(); ++i) {
            std::cout << action_params[i];
            if (i < action_params.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        return {action, action_params};
    }

    void append_sample(const PDQNExperience& e) {
        if (memory.size() >= memory_maxlen)
            memory.pop_front();
        memory.push_back(e);
    }

    void train_model() {
        if (epsilon > epsilon_min)
            epsilon *= epsilon_decay;
        else
            epsilon = epsilon_min;

        std::vector<PDQNExperience> batch;
        std::sample(memory.begin(), memory.end(), std::back_inserter(batch), batch_size, std::mt19937{std::random_device{}()});

        std::vector<float> rewards, actions, dones;
        std::vector<std::vector<float>> states, next_states, agent_inputs, next_agent_inputs;
        for (const auto& e : batch) {
            states.push_back(e.state);
            actions.push_back(e.action);
            rewards.push_back(e.reward);
            next_states.push_back(e.next_state);
            dones.push_back(e.done ? 1.0f : 0.0f);

            // Concatenate state + param
            std::vector<float> input = e.state;
            input.insert(input.end(), e.action_params.begin(), e.action_params.end());
            agent_inputs.push_back(input);

            // Predict next parameters from target actor
            torch::Tensor next_state_tensor = torch::tensor(e.next_state, torch::kFloat32);
            torch::Tensor pred_action_param_tensor = target_actor_model.forward(next_state_tensor.unsqueeze(0)).detach().squeeze(0);
            std::vector<float> pred_action_params(
                pred_action_param_tensor.data_ptr<float>(),
                pred_action_param_tensor.data_ptr<float>() + pred_action_param_tensor.numel()
            );

            // Concatenate next state + predicted params
            std::vector<float> next_input = e.next_state;
            next_input.insert(next_input.end(), pred_action_params.begin(), pred_action_params.end());
            next_agent_inputs.push_back(next_input);
        }

        std::vector<torch::Tensor> tensor_seq;
        for (const auto& v : agent_inputs)
            tensor_seq.push_back(torch::tensor(v, torch::TensorOptions().dtype(torch::kFloat)));
        auto input_tensor = torch::stack(tensor_seq);

        auto q_values = q_model.forward(input_tensor);
        auto action_tensor = torch::tensor(actions, torch::kLong).unsqueeze(1);
        auto reward_tensor = torch::tensor(rewards);
        auto done_tensor = torch::tensor(dones);
        auto q_selected = q_values.gather(1, action_tensor).squeeze();

        std::vector<torch::Tensor> next_tensor_seq;
        for (const auto& v : next_agent_inputs)
            next_tensor_seq.push_back(torch::tensor(v, torch::TensorOptions().dtype(torch::kFloat)));
        auto next_input_tensor = torch::stack(next_tensor_seq);

        torch::Tensor max_next_q;
        {
            torch::NoGradGuard no_grad;
            auto next_q_values = target_q_model.forward(next_input_tensor);
            max_next_q = std::get<0>(next_q_values.max(1));
        }

        auto target = reward_tensor + discount_factor * max_next_q * (1 - done_tensor);
        auto critic_loss = torch::mse_loss(q_selected, target.detach());
        q_optimizer.zero_grad();
        critic_loss.backward();
        q_optimizer.step();

        std::vector<torch::Tensor> state_tensors;
        for (const auto& s : states)
            state_tensors.push_back(torch::tensor(s, torch::TensorOptions().dtype(torch::kFloat)));
        auto state_batch = torch::stack(state_tensors);

        auto action_param_batch = actor_model.forward(state_batch);

        std::vector<torch::Tensor> q_input_tensors;
        for (int i = 0; i < state_batch.size(0); ++i)
            q_input_tensors.push_back(torch::cat({state_batch[i], action_param_batch[i]}, 0));
        auto q_input_batch = torch::stack(q_input_tensors).detach();

        auto q_eval = q_model.forward(q_input_batch);
        auto max_q = std::get<0>(q_eval.max(1));
        auto actor_loss = -max_q.mean();

        actor_optimizer.zero_grad();
        actor_loss.backward();
        torch::nn::utils::clip_grad_norm_(actor_model.parameters(), 10.0);
        actor_optimizer.step();

        std::cout << "[Train] Critic Loss: " << critic_loss.item<float>()
                  << ", Actor Loss: " << actor_loss.item<float>() << std::endl;
    }

    void step(std::vector<float> state, 
              int action, 
              std::vector<float> action_param, 
              float reward, 
              std::vector<float> next_state, 
              int done,
              bool is_training) {
        PDQNExperience e = {state, action, action_param, reward, next_state, done};
        append_sample(e);

        if (memory.size() >= train_start && memory.size() > batch_size) {
            train_model();
            update_target_model();
        }
    }

    void save_model(const std::string& directory) {
        // Ensure the directory exists
        if (!fs::exists(directory)) {
            fs::create_directories(directory);
        }

        // Save actor model
        torch::serialize::OutputArchive actor_archive;
        actor_model.save(actor_archive);
        std::string actor_path = directory + "/pdqn_actor_model.pt";
        actor_archive.save_to(actor_path);
        std::cout << "ParamActor model saved to: " << actor_path << std::endl;

        // Save q model
        torch::serialize::OutputArchive q_archive;
        q_model.save(q_archive);
        std::string q_model_path = directory + "/pdqn_q_model.pt";
        q_archive.save_to(q_model_path);
        std::cout << "Q_CriticNetwork model saved to: " << q_model_path << std::endl;
    }

    void load_model(const std::string& directory) {
        // Load actor model
        std::string actor_path = directory + "/pdqn_actor_model.pt";
        try {
            torch::serialize::InputArchive actor_archive;
            actor_archive.load_from(actor_path);
            actor_model.load(actor_archive);
            std::cout << "Actor model loaded from: " << actor_path << std::endl;
        } catch (const c10::Error& e) {
            std::cerr << "Error loading actor model: " << e.what() << std::endl;
        }

        // Load Q model
        std::string q_model_path = directory + "/pdqn_q_model.pt";
        try {
            torch::serialize::InputArchive q_archive;
            q_archive.load_from(q_model_path);
            q_model.load(q_archive);
            std::cout << "Q model loaded from: " << q_model_path << std::endl;
        } catch (const c10::Error& e) {
            std::cerr << "Error loading Q model: " << e.what() << std::endl;
        }
    }

private:
    std::mt19937 rng;
    int state_size, action_size, action_parameter_size;
    float epsilon, epsilon_min, epsilon_decay, discount_factor;
    size_t memory_maxlen, batch_size, train_start;
    float tau_q, tau_actor;
    std::deque<PDQNExperience> memory;

    ParamActor actor_model, target_actor_model;
    Q_CriticNetwork q_model, target_q_model;
    torch::optim::Adam actor_optimizer, q_optimizer;
};

#endif // DRQN_AGENT_H