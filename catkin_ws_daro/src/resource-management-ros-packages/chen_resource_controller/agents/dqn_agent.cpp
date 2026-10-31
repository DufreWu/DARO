#include "dqn_agent.h"
#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;

DQNAgent::DQNAgent(int state_size, int action_size)
: state_size(state_size), 
  action_size(action_size),
  qnetwork_local(state_size, action_size),
  qnetwork_target(state_size, action_size),
  optimizer(qnetwork_local.parameters(), torch::optim::AdamOptions(1e-3))
   {
    torch::NoGradGuard no_grad;
    for (size_t i = 0; i < qnetwork_local.parameters().size(); ++i) {
        qnetwork_target.parameters()[i].copy_(qnetwork_local.parameters()[i]);
    }

    epsilon = 1.0f;
    epsilon_min = 0.1f;
    epsilon_decay = 0.95f;
    discount_factor = 0.99f;
}

int DQNAgent::act(std::vector<float> state) {
    torch::Tensor state_tensor = torch::tensor(state).unsqueeze(0);
    qnetwork_local.eval();
    torch::NoGradGuard no_grad;
    auto q_values = qnetwork_local.forward(state_tensor);
    qnetwork_local.train();

    if (((float) rand() / RAND_MAX) > epsilon) {
        return q_values.argmax(1).item<int>();  // greedy action
    } else {
        return rand() % action_size;            // random action
    }
}

void DQNAgent::step(std::vector<float> state, int action, float reward, std::vector<float> next_state, bool done) {
    auto s  = torch::tensor(state, torch::dtype(torch::kFloat));
    auto a  = torch::tensor({action}, torch::dtype(torch::kLong));
    auto r  = torch::tensor({reward});
    auto ns = torch::tensor(next_state, torch::dtype(torch::kFloat));
    auto d  = torch::tensor({done ? 1.0f : 0.0f});

    // Store experience
    memory.push_back(std::make_tuple(s, a, r, ns, d));
    if (memory.size() > memory_size)
        memory.pop_front();

    // Learn only if enough samples are collected
    if (memory.size() > batch_size) {
        std::vector<DQNExperience> experiences;

        std::mt19937 gen(std::random_device{}());
        std::uniform_int_distribution<> dist(0, memory.size() - 1);

        for (int i = 0; i < batch_size; ++i) {
            experiences.push_back(memory[dist(gen)]);
        }

        std::cout << "[DQNAgent] Learning from memory..." << std::endl;
        learn(experiences, gamma);
    }
}


void DQNAgent::learn(std::vector<DQNExperience> experiences, float gamma) {
    // Unpack experiences
    std::vector<torch::Tensor> states, actions, rewards, next_states, dones;
    for (const auto& [s, a, r, ns, d] : experiences) {
        states.push_back(s);
        actions.push_back(a);
        rewards.push_back(r);
        next_states.push_back(ns);
        dones.push_back(d);
    }

    // Stack into batched tensors
    auto state_batch = torch::stack(states);        // [B, state_dim]
    auto action_batch = torch::stack(actions);      // [B, 1]
    auto reward_batch = torch::stack(rewards);      // [B, 1]
    auto next_state_batch = torch::stack(next_states);
    auto done_batch = torch::stack(dones);          // [B, 1]

    // Compute max Q value for next state from target model
    auto max_result = qnetwork_target.forward(next_state_batch).max(1, true);
    auto Q_targets_next = std::get<0>(max_result);  // [B, 1]

    // Bellman equation for target Q values
    auto Q_targets = reward_batch + (gamma * Q_targets_next * (1 - done_batch));

    // Expected Q values from local model
    auto Q_expected = qnetwork_local.forward(state_batch).gather(1, action_batch);

    // Compute and optimize loss
    auto loss = torch::mse_loss(Q_expected, Q_targets);
    optimizer.zero_grad();
    loss.backward();
    optimizer.step();

    // --------------------- Soft update target network ---------------------
    torch::NoGradGuard no_grad;
    for (size_t i = 0; i < qnetwork_local.parameters().size(); ++i) {
        qnetwork_target.parameters()[i].copy_(
            tau * qnetwork_local.parameters()[i] + (1.0 - tau) * qnetwork_target.parameters()[i]
        );
    }

    // Epsilon decay
    if (epsilon > epsilon_min)
        epsilon *= epsilon_decay;
}

void DQNAgent::save_model(const std::string& directory) {
    // Ensure the directory exists
    if (!fs::exists(directory)) {
        fs::create_directories(directory);
    }

    // Save actorcritic model
    torch::serialize::OutputArchive dqn_archive;
    qnetwork_local.save(dqn_archive);
    std::string dqn_path = directory + "/dqn_model.pt";
    dqn_archive.save_to(dqn_path);
    std::cout << "DQN model saved to: " << dqn_path << std::endl;
}

bool DQNAgent::load_model(const std::string& directory) {
    try {
        // Load policy
        {
            torch::serialize::InputArchive ar;
            ar.load_from(directory + "/dqn_model.pt");
            qnetwork_local.load(ar);
        }
        return true;
    } catch (const std::exception& ex) {
        std::cerr << "Failed to load DQN models: " << ex.what() << std::endl;
        return false;
    }
}
