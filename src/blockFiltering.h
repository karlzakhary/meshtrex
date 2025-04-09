#pragma once

struct MinMaxResult {
    uint32_t minVal;
    uint32_t maxVal;
};

int filterUnoccupiedBlocks(char **argv, const char *path);