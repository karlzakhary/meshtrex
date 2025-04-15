#version 450
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require // For subgroupAdd
#extension GL_KHR_shader_subgroup_ballot : require // For subgroupBallot, subgroupExclusiveAdd etc. (alternative scan)
// #extension GL_NV_shader_atomic_int64 : require // If using 64-bit atomics/prefix sum
// #extension GL_KHR_shader_subgroup_shuffle : require // Needed for paper's shuffle-scan

// --- Push Constants / Uniforms ---
layout(push_constant) uniform PushConstants {
    uvec4 volumeDim;    // Actual dimensions of the input volume texture
    uvec4 blockDim;     // Should match local_size (e.g., 8,8,8)
    uvec4 blockGridDim; // Dimensions of the dispatch grid (and the output image)
    float isovalue;     // The target isovalue
} pc;

// --- Bindings ---

// Binding 0: Input Min/Max data (produced by previous pass)
// Assumes image texture format matching previous step (rg32ui -> uvec2)
layout(binding = 0, rg32ui) uniform readonly uimage3D minMaxInputVolume;

// Binding 1: Output buffer for compacted active block IDs
layout(binding = 1, std430) buffer CompactedBlockIDs {
    uint blockIDs[]; // Stores 1D indices of active blocks
};

// Binding 2: Atomic counter for total active blocks & output offset calculation
layout(binding = 2, std430) buffer ActiveBlockCount {
    uint count; // Stores the total number of active blocks found
} activeBlockCount;

// --- Workgroup Setup ---
layout (local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

// Shared memory for intra-workgroup communication/scan if needed
// shared uint s_workgroup_scan[gl_WorkGroupSize.x / gl_SubgroupSize];

void main() {
    // Calculate the 1D index of the block this invocation is responsible for
    uint globalBlockID1D = gl_GlobalInvocationID.x;

    // Calculate total number of blocks in the original grid
    uint totalBlocks = pc.blockGridDim.x * pc.blockGridDim.y * pc.blockGridDim.z;

    // --- Bounds Check ---
    if (globalBlockID1D >= totalBlocks) {
        return;
    }

    // --- Determine if Block is Active ---
    // Calculate 3D block coordinates from 1D index
    uvec3 blockCoord;
    uint planeSize = pc.blockGridDim.x * pc.blockGridDim.y;
    blockCoord.z = (planeSize > 0) ? (globalBlockID1D / planeSize) : 0;
    uint remainder = (planeSize > 0) ? (globalBlockID1D % planeSize) : globalBlockID1D;
    blockCoord.y = (pc.blockGridDim.x > 0) ? (remainder / pc.blockGridDim.x) : 0;
    blockCoord.x = (pc.blockGridDim.x > 0) ? (remainder % pc.blockGridDim.x) : remainder;


    // Read min/max for this block (assuming uint32 values from rg32ui)
    uvec2 minMax = imageLoad(minMaxInputVolume, ivec3(blockCoord)).xy;

    // --- REVISED ACTIVITY CHECK ---
    // Check if the isovalue falls within the [min, max] range of the block.
    // IMPORTANT: Ensure 'pc.isovalue' is scaled to the same range as the data
    // represented by minMax.x and minMax.y (e.g., 0-255 if original data was uint8).
    // If isovalue is normalized [0,1], normalize minMax or scale isovalue.
    // Using <= maxVal is generally needed to catch surfaces touching the max value.
    // Also check minVal != maxVal to exclude blocks with constant value equal to isovalue (optional but common).

    bool blockIsActive = false;
    // Check if min and max are different to avoid trivial blocks (optional)
    if (minMax.x != minMax.y) {
        // Check if isovalue is within the range [min, max]
        blockIsActive = (pc.isovalue >= float(minMax.x) && pc.isovalue <= float(minMax.y));
    }
    // If you want to include blocks where min==max==isovalue, use:
    // blockIsActive = (pc.isovalue >= float(minMax.x) && pc.isovalue <= float(minMax.y));

    uint activeFlag = blockIsActive ? 1 : 0;

    // --- Stream Compaction / Prefix Sum ---
    // Calculate the output offset for this block if it's active.
    // Simplified placeholder using atomicAdd - DOES NOT PRESERVE ORDER.
    // A full parallel scan is needed for ordered, contiguous output.
    uint outputOffset = 0;
    if (activeFlag == 1) {
        // Atomically increment the global counter and get the value *before* incrementing.
        outputOffset = atomicAdd(activeBlockCount.count, 1);
    }

    // --- Write Output ---
    // If this block is active, write its 1D ID to the output buffer at the calculated offset.
    if (activeFlag == 1) {
        // Bounds check on output offset might be needed if buffer size is strictly limited
        // (though typically it's sized for the worst case = totalBlocks)
        blockIDs[outputOffset] = globalBlockID1D;
    }

    // NOTE: The simplified atomic approach above does not guarantee order.
    // A full parallel scan implementation is required for ordered output.
    // The final value in ActiveBlockCount.count will be correct.
}
