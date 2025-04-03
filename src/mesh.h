#pragma once

#include "glmMath.h"

struct Vertex {
    float vx, vy, vz;
    uint8_t nx, ny, nz, nw;
    float tu, tv;
};

struct alignas(16) Mesh {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
};

bool loadMesh(Mesh& result, const char *path);