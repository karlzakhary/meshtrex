#include <vector>
#ifndef __APPLE__
#include <cstdint>
#endif
#include "blockFiltering.h"
#include "testMinMax.h"

int main(int argc, char* argv[]) {
    uint32_t gpuActiveCount = filterUnoccupiedBlocks(
       argv, "raw_volumes/bonsai_256x256x256_uint8.raw"); // Placeholder - replace with actual GPU result
    return 0;
}
