cmake_minimum_required(VERSION 3.30)

project(meshtrex)

file(GLOB_RECURSE GLSL_SOURCE_FILES "src/shaders/*.glsl")
file(GLOB_RECURSE GLSL_HEADER_FILES "src/shaders/*.h" "src/config.h")
file(GLOB_RECURSE CPP_SOURCE_FILES "src/*.h" "src/*.cpp")

add_executable(meshtrex
        ${CPP_SOURCE_FILES}
        ${GLSL_SOURCE_FILES}
        ${GLSL_HEADER_FILES})

set_target_properties(meshtrex PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)

target_compile_definitions(meshtrex PRIVATE GLFW_INCLUDE_NONE GLM_FORCE_XYZW_ONLY GLM_FORCE_QUAT_DATA_XYZW GLM_FORCE_QUAT_CTOR_XYZW)
target_include_directories(meshtrex PRIVATE external/fast_obj)

if(APPLE)
    target_compile_definitions(meshtrex PRIVATE VK_USE_PLATFORM_METAL_EXT)
    target_compile_definitions(meshtrex PRIVATE GLFW_EXPOSE_NATIVE_COCOA)
    set(VOLK_STATIC_DEFINES VK_USE_PLATFORM_METAL_EXT)
elseif(UNIX)
    target_compile_definitions(meshtrex PRIVATE VK_USE_PLATFORM_XLIB_KHR)
    target_compile_definitions(meshtrex PRIVATE GLFW_EXPOSE_NATIVE_X11)
    set(VOLK_STATIC_DEFINES VK_USE_PLATFORM_XLIB_KHR)
endif()

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

add_subdirectory(external/glfw)
add_subdirectory(external/glm)
add_subdirectory(external/volk)
add_subdirectory(external/meshoptimizer)

add_library(imgui STATIC external/imgui/imgui.cpp external/imgui/imgui_draw.cpp external/imgui/imgui_widgets.cpp external/imgui/imgui_tables.cpp external/imgui/imgui_demo.cpp external/imgui/backends/imgui_impl_vulkan.cpp external/imgui/backends/imgui_impl_glfw.cpp)
target_include_directories(imgui PUBLIC external/imgui external/imgui/backends)
set_target_properties(imgui PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)
target_compile_definitions(imgui PRIVATE IMGUI_IMPL_VULKAN_USE_VOLK)
target_link_libraries(imgui PRIVATE volk glfw)

target_link_libraries(meshtrex
        PRIVATE
        glfw
        glm::glm
        volk
        meshoptimizer
        imgui)

if(DEFINED ENV{VULKAN_SDK})
    set(GLSL_VALIDATOR "$ENV{VULKAN_SDK}/bin/glslangValidator")
else()
    set(GLSL_VALIDATOR "glslangValidator")
endif()

# Thanks to: https://gist.github.com/evilactually/a0d191701cb48f157b05be7f74d79396
set(SPIRV_OUTPUT_DIR "${PROJECT_BINARY_DIR}/spirv/")
foreach(GLSL ${GLSL_SOURCE_FILES})
    get_filename_component(STEM ${GLSL} NAME_WLE)
    set(SPIRV "${SPIRV_OUTPUT_DIR}${STEM}.spv")
    add_custom_command(
            OUTPUT ${SPIRV}
            COMMAND ${CMAKE_COMMAND} -E make_directory "${SPIRV_OUTPUT_DIR}"
            COMMAND ${GLSL_VALIDATOR} -V --target-env vulkan1.2 --quiet ${GLSL} -o ${SPIRV}
            DEPENDS ${GLSL} ${GLSL_HEADER_FILES})
    list(APPEND SPIRV_BINARY_FILES ${SPIRV})
endforeach()

add_custom_target(compile_shaders DEPENDS ${SPIRV_BINARY_FILES})
add_dependencies(meshtrex compile_shaders)

add_custom_command(
        TARGET meshtrex POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/meshes
        ${CMAKE_BINARY_DIR}/meshes
        COMMENT "Copying meshes folder to build directory"
)

add_custom_command(
        TARGET meshtrex POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/raw_volumes
        ${CMAKE_BINARY_DIR}/raw_volumes
        COMMENT "Copying meshes folder to build directory"
)