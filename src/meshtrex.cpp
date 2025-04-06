#include <vector>

#include "blockFiltering.h"

int main(int argc, char* argv[]) {
    // Compute Min/Max for all blocks by calling the single function
    std::vector<MinMaxResult> gpudata = filterUnoccupiedBlocks(
        argv, "raw_volumes/bonsai_256x256x256_uint8.raw");

    return 0;
}