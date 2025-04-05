#include <iostream>

#include "blockFiltering.h"

int main(int argc, char **argv)
{
    filterUnoccupiedBlocks(argv, "raw_volumes/aneurism_256x256x256_uint8.raw");
    std::cout << "Hello, World!" << std::endl;
    return 0;
}