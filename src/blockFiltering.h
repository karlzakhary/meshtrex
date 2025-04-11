#pragma once

struct MinMaxResult {
    uint32_t minVal;
    uint32_t maxVal;
};

uint32_t filterUnoccupiedBlocks(char **argv, const char *path);