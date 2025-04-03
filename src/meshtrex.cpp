#include <iostream>

#include "objViewer.h"

int main(int argc, char **argv)
{
    drawObject(argv, "meshes/kitten.obj");
    std::cout << "Hello, World!" << std::endl;
    return 0;
}