# MeshTrex

GPU-accelerated real-time isosurface extraction using Vulkan mesh shaders and Marching Cubes.

## Overview

MeshTrex extracts isosurfaces from volumetric datasets (medical imaging, scientific simulation, engineering) at interactive frame rates on the GPU. It implements the Parallel Marching Blocks (PMB) algorithm mapped onto Vulkan task/mesh shaders, bypassing the traditional vertex pipeline entirely. A hierarchical min-max octree, stream compaction, and occlusion culling reduce work to only the blocks that contribute visible geometry each frame. This project was developed as part of a master's thesis on GPU-accelerated isosurface extraction.

## Features

- **Parallel Marching Blocks (PMB)** -- meshlet-based isosurface extraction via task/mesh shaders with edge-ownership deduplication
- **Min-max octree acceleration** -- hierarchical empty-space skipping to avoid processing blocks with no surface
- **Occlusion culling** -- both Hi-Z pyramid (compute) and rasterization-based approaches
- **Temporal coherence** -- reuses the previous frame's Potentially Visible Set (PVS), only extracting newly visible blocks
- **Transient extraction** -- on-the-fly geometry generation for dynamic isovalue changes without persistent storage
- **Stream compaction** -- atomic-based active block filtering (2.4x faster than prefix sum on Ampere)
- **Interactive controls** -- ImGui interface for isovalue adjustment, rendering parameters, and profiling
- **Cross-platform** -- Linux, Windows, and macOS (via MoltenVK/Metal translation)

## Pipeline

```
Volume Data (3D Texture)
       |
       v
1. Min-Max Octree Generation    [minMaxLeaf.comp, minMaxOctreeReduce.comp]
       |
       v
2. Active Block Filtering        [occupiedBlockFiltering.comp]
       |
       v
3. Occlusion Culling             [compute_occlusion_culling.comp / raster_occlusion.task+mesh]
       |
       v
4. Mesh Extraction               [marching_cubes_pmb.task + marching_cubes_pmb.mesh]
       |
       v
5. Rendering                     [render.task + render.mesh + render.frag]
```

## Prerequisites

| Requirement | Version |
|---|---|
| Vulkan SDK | 1.3+ (1.2 on macOS) |
| CMake | 3.28.3+ |
| C++ compiler | C++20 (GCC 12+, Clang 15+, MSVC 2022+) |
| `glslangValidator` | Included in Vulkan SDK |
| GPU | NVIDIA Turing or newer recommended (mesh shader support) |

## Building

Clone with submodules and build:

```bash
git clone --recursive https://github.com/karlzakhary/meshtrex.git
cd meshtrex
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

For a debug build with validation layers and shader debug output:

```bash
cmake -DCMAKE_BUILD_TYPE=Debug ..
make -j$(nproc)
```

**macOS note:** MoltenVK translates Vulkan calls to Metal. The effective API level is Vulkan 1.2, and mesh shaders are not available -- the pipeline falls back to compute-based extraction.

## Usage

```
./meshtrex [OPTIONS]
```

| Option | Description |
|---|---|
| `--volume <path>` | Path to a raw volume file |
| `--isovalue <value>` | Isovalue threshold (default: 80.0) |
| `--transient` | Transient extraction mode (on-the-fly geometry, no persistent storage) |
| `--temporal` | Enable temporal coherence optimization |
| `--synthetic [type] [count]` | Use synthetic test volume (`random`, `layered`, `stress`) |
| `--benchmark` | Run automated benchmark mode |
| `--viewpoint <num>` | Specific viewpoint for benchmark |
| `--density-dispatch` | Enable density-based dispatch |
| `--collect-metrics` | Collect occlusion metrics |
| `--help` | Show help |

### Example

```bash
./meshtrex --volume raw_volumes/aneurism_256x256x256_uint8.raw --isovalue 80 --temporal
```

### Controls

- **W/A/S/D/E/Q** -- camera movement
- **Mouse** -- camera rotation
- **ImGui panel** -- isovalue slider, rendering options, profiling stats

### Tested Volume Datasets

| Volume | Resolution | Format |
|---|---|---|
| `aneurism` | 256x256x256 | uint8 |
| `bonsai` | 256x256x256 | uint8 |
| `chameleon` | 1024x1024x1080 | uint16 |
| `csafe_heptane` | 302x302x302 | uint8 |
| `kingsnake` | 1024x1024x795 | uint8 |
| `magnetic_reconnection` | 512x512x512 | float32 |
| `marmoset_neurons` | 1024x1024x314 | uint8 |

## Project Structure

```
meshtrex/
  src/
    meshtrex.cpp                  # Entry point, CLI parsing
    renderingManager.cpp/h        # Window, swapchain, main render loop
    minMaxManager.cpp/h           # Min-max octree generation
    filteringManager.cpp/h        # Active block stream compaction
    extractionManager.cpp/h       # Persistent mesh extraction
    transientExtractionManager.cpp/h  # Transient extraction mode
    computeOcclusionPass.cpp/h    # Hi-Z pyramid occlusion culling
    rasterOcclusionPass.cpp/h     # Rasterization-based occlusion culling
    device.cpp/h                  # Vulkan device setup and feature detection
    vulkan_context.cpp/h          # Vulkan instance and surface
    gpuProfiler.h                 # GPU timestamp profiling
    profilingManager.h            # CPU/GPU metrics aggregation
    shaders/                      # GLSL compute, task, mesh, fragment shaders
  external/                       # Git submodule dependencies
    glfw/                         # Windowing and input
    glm/                          # Math library
    volk/                         # Vulkan meta-loader
    imgui/                        # Immediate-mode GUI
    meshoptimizer/                # Mesh optimization
    fast_obj/                     # OBJ file loading
  raw_volumes/                    # Volume datasets (not tracked in git)
  thesis/                         # LaTeX thesis source
  docs/                           # Architecture docs and references
```

## Acknowledgments

This project was developed as part of a master's thesis on GPU-accelerated isosurface extraction. The Parallel Marching Blocks algorithm is based on the work of Liu et al. Occlusion culling approaches draw on Hi-Z culling (Greene et al.) and rasterization-based culling (Kreskowski).
