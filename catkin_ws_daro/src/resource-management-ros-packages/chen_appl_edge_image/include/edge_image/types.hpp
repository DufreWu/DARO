#ifndef EDGE_IMAGE_TYPES_HPP
#define EDGE_IMAGE_TYPES_HPP

// A simple POD (Plain Old Data) struct that both C++ and CUDA can use.
struct GpuPoint2i {
    int x;
    int y;
};

#endif