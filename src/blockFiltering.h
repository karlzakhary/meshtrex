#pragma once

struct MinMaxResult {
    uint32_t minVal;
    uint32_t maxVal;
};

std::vector<MinMaxResult> filterUnoccupiedBlocks(char **argv, const char *path);