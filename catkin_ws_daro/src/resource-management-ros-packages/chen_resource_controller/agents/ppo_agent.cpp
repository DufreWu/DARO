#include "ppo_agent.h"
#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;

PPOAgent::PPOAgent(int state_size, int action_size)
    : model(std::make_shared<ActorCritic>(state_size, action_size)),
      optimizer(model->parameters(), torch::optim::AdamOptions(3e-4)),
      state_size(state_size), action_size(action_size) {
        gamma = 0.99;
        clip_param = 0.2;
      }

int PPOAgent::act(std::vector<float> state) {
    torch::NoGradGuard no_grad;
    torch::Tensor state_tensor = torch::tensor(state).unsqueeze(0);
    auto [logits, value] = model->forward(state_tensor);
    auto probs = torch::softmax(logits, -1);
    auto action = torch::multinomial(probs, 1);
    auto log_prob = torch::log(probs.gather(1, action));

    Trajectory traj;
    traj.state = state_tensor;
    traj.action = action.squeeze(1);
    traj.log_prob = log_prob;
    traj.value = value;
    traj.reward = torch::zeros({1});
    traj.done = torch::zeros({1});
    memory.push_back(traj);

    return action.item().toInt();
}

void PPOAgent::store(const Trajectory& traj) {
    memory.back().reward = traj.reward;
    memory.back().done = traj.done;
}

void PPOAgent::learn() {
    const int K_EPOCH = 4;
    const float eps = 1e-8;

    // Compute discounted returns
    std::vector<torch::Tensor> returns;
    torch::Tensor G = torch::zeros({1});
    for (auto it = memory.rbegin(); it != memory.rend(); ++it) {
        G = it->reward + gamma * G * (1 - it->done);
        returns.insert(returns.begin(), G.detach());
    }

    // Gather tensors from memory
    std::vector<torch::Tensor> states, actions, old_log_probs, values;
    for (const auto& traj : memory) {
        states.push_back(traj.state);
        actions.push_back(traj.action);
        old_log_probs.push_back(traj.log_prob);
        values.push_back(traj.value);
    }

    auto states_tensor = torch::cat(states);
    auto actions_tensor = torch::cat(actions).unsqueeze(1);
    auto old_log_probs_tensor = torch::cat(old_log_probs);
    auto values_tensor = torch::cat(values);
    auto returns_tensor = torch::cat(returns);

    // Normalize advantage
    auto advantage = returns_tensor - values_tensor;
    advantage = (advantage - advantage.mean()) / (advantage.std() + eps);

    for (int i = 0; i < K_EPOCH; ++i) {
        auto [logits, new_values] = model->forward(states_tensor);
        auto probs = torch::softmax(logits, -1);
        auto new_log_probs = torch::log(probs.gather(1, actions_tensor));

        auto ratio = torch::exp(new_log_probs - old_log_probs_tensor);
        auto surr1 = ratio * advantage;
        auto surr2 = torch::clamp(ratio, 1 - clip_param, 1 + clip_param) * advantage;
        auto actor_loss = -torch::min(surr1, surr2).mean();
        auto critic_loss = torch::nn::functional::mse_loss(new_values, returns_tensor);

        // Optional: add entropy bonus for exploration
        auto entropy = -(probs * torch::log(probs + eps)).sum(-1).mean();

        auto loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy;

        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
    }

    memory.clear();
}

void PPOAgent::step(const std::vector<float>& next_state,
                    float reward,
                    bool done) {
    // Update reward and done in last trajectory step
    if (!memory.empty()) {
        memory.back().reward = torch::tensor({reward});
        memory.back().done = torch::tensor({done ? 1.0f : 0.0f});
    }

    // Trigger learning if trajectory is long enough or episode is done
    
    if (memory.size() > rollout_length || done) {
        learn();  // PPO training step
    }
}

void PPOAgent::save_model(const std::string& directory) {
    // Ensure the directory exists
    if (!fs::exists(directory)) {
        fs::create_directories(directory);
    }

    // Save actorcritic model
    torch::serialize::OutputArchive ppo_archive;
    model->save(ppo_archive);
    std::string ppo_path = directory + "/ppo_model.pt";
    ppo_archive.save_to(ppo_path);
    std::cout << "PPO model saved to: " << ppo_path << std::endl;
}

