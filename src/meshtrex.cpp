#include "common.h"

#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
#include <fast_obj.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#include <meshoptimizer.h>
#include <string.h>

#include <iostream>
#include <vector>

#include "config.h"
#include "device.h"
#include "glm/vec4.hpp"
#include "math.h"
#include "resources.h"
#include "shaders.h"
#include "swapchain.h"
#include "vulkan_utils.h"
#include "triangle.cpp"

void execute(char** argv);

int main(int argc, char** argv)
{
    execute(argv);
    std::cout << "Hello, World!" << std::endl;
    return 0;
}