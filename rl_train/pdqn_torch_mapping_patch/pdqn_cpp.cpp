#include <torch/script.h>
#include <iostream>
#include <vector>
#include <utility> // For std::pair
#include <cstring>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <sstream> // For std::ostringstream

#define PORT 8702

// Define CPU and GPU frequency lists
std::vector<int> cpu_fre_list = {499200, 652800, 806400, 960000, 1113600, 1267200, 1420800, 1574400, 1728000, 1881600, 2035200};
std::vector<int> gpu_fre_list = {229500000, 318750000, 408000000, 497250000, 586500000, 675750000, 765000000, 854250000,
                                 943500000, 1032750000, 1122000000, 1211250000, 1300500000};

// Generate all combinations of CPU and GPU frequency indices
std::vector<std::pair<int, int>> clk_action_list;

void initialize_clk_action_list() {
    for (int i = 0; i < cpu_fre_list.size(); ++i) {
        for (int j = 0; j < gpu_fre_list.size(); ++j) {
            clk_action_list.emplace_back(i, j); // Add the pair (i, j) to the list
        }
    }
}

int main() {
    // Initialize clk_action_list
    initialize_clk_action_list();

    int sock = 0;
    struct sockaddr_in serv_addr;
    const char* message = "Hello from client";
    char buffer[1024] = {0};


    // Create socket file descriptor
    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        std::cerr << "Socket creation error" << std::endl;
        return -1;
    }

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(PORT);

    // Convert IPv4 address from text to binary form
    if (inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr) <= 0) {
        std::cerr << "Invalid address or address not supported" << std::endl;
        return -1;
    }

    // Connect to the server
    if (connect(sock, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) < 0) {
        std::cerr << "Connection failed" << std::endl;
        return -1;
    }

    // Load the TorchScript model
    torch::jit::script::Module q_module;
    torch::jit::script::Module actor_module;
    try {
        q_module = torch::jit::load("./../save_model/model_Q_torch.pt");
        actor_module = torch::jit::load("./../save_model/model_actor_torch.pt");
    } catch (const c10::Error& e) {
        std::cerr << "Model loading failed.\n";
        return -1;
    }

    // Example input state
    std::vector<float> state = {0, 8, 9.458, 20.462164, 79.96999526}; // Use your actual dimensions
    torch::Tensor state_tensor = torch::from_blob(state.data(), {1, state.size()}).clone();

    // Serialize the state vector into a comma-separated string
    // std::vector<float> state = {0, 8, 9.458, 20.462164, 79.96999526}; // Use your actual dimensions
    std::ostringstream state_stream;
    for (size_t i = 0; i < state.size(); ++i) {
        state_stream << state[i];
        if (i != state.size() - 1) {
            state_stream << ","; // Add a comma between elements
        }
    }
    std::string state_message = state_stream.str(); // Convert the stream to a string

    while (1) {
        // Send the serialized state message over the socket
        send(sock, state_message.c_str(), state_message.size(), 0);
        std::cout << "State message sent: " << state_message << std::endl;

        // Read the server's response
        read(sock, buffer, 1024);
        std::cout << "Message from server: " << buffer << std::endl;
    }

    // Measure actor inference time
    auto start_actor = std::chrono::high_resolution_clock::now();
    torch::Tensor action_param = actor_module.forward({state_tensor}).toTensor();
    auto end_actor = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> actor_inference_time = end_actor - start_actor;
    std::cout << "Actor inference time: " << actor_inference_time.count() << " seconds" << std::endl;

    // Concatenate state + action_param
    torch::Tensor q_input = torch::cat({state_tensor, action_param}, 1);

    // Measure Q-network inference time
    auto start_q = std::chrono::high_resolution_clock::now();
    torch::Tensor q_values = q_module.forward({q_input}).toTensor();
    auto end_q = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> q_inference_time = end_q - start_q;
    std::cout << "Q-network inference time: " << q_inference_time.count() << " seconds" << std::endl;

    // Get the best action
    int best_action = q_values.argmax(1).item<int>();
    int cpu_freq_index = clk_action_list[best_action].first;
    int gpu_freq_index = clk_action_list[best_action].second;
    std::cout << "Best action_param: " << action_param << std::endl;
    std::cout << "Best action: " << best_action << 
                 ", cpu_freq_index: " << cpu_freq_index << 
                 ", gpu_freq_index: " << gpu_freq_index << std::endl;

    // Close the socket
    close(sock);
}
