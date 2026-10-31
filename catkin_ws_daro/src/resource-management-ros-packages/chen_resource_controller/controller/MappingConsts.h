#ifndef MAPPING_CONSTANTS_H_
#define MAPPING_CONSTANTS_H_

#include <map>

#define NUM_OF_CORES          3
#define NUM_OF_APPLICATIONS   5

enum {
  MAP_LITTLE_ID = 0,
  MAP_BIG_ID = 1,
  MAP_GPU_ID = 2,
  MAP_MAX_ID = 3
};

const std::map<std::string, int> appls_index_map = {
    {"edge_image",         0},
    {"denoising_filling",  1},
    {"distance_surface",   2},
    {"optical_flow",       3},
    {"tracker",            4}
};

extern std::vector<int> appls_to_cluster;

#endif
