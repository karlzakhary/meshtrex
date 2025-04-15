#version 450

// --- Push Constants / Uniforms ---
// Uses uint isovalue and uvec4 for vec3 alignment
layout(push_constant) uniform PushConstants {
    uvec4 volumeDim;
    uvec4 blockDim;
    uvec4 blockGridDim;
    float isovalue;
} pc;

// --- Bindings ---

// Binding 0: Input Min/Max data image (rg32ui -> uvec2)
layout(binding = 0, rg32ui) uniform readonly uimage3D minMaxInputVolume;

// Binding 1: Output buffer for compacted active block IDs
layout(binding = 1, std430) buffer CompactedBlockIDs {
    uint blockIDs[]; // Stores 1D indices of active blocks
};

// Binding 2: Atomic counter for total active blocks & output offset calculation
layout(binding = 2, std430) buffer ActiveBlockCount {
    uint count; // Host initializes to 0. Stores total active count.
} activeBlockCount; // Added instance name

// --- Workgroup Setup ---
// Workgroup size used for the scan (e.g., 128)
#define WORKGROUP_SIZE 128
layout (local_size_x = WORKGROUP_SIZE, local_size_y = 1, local_size_z = 1) in;

// --- Shared Memory ---
// Data for the scan algorithm
shared uint s_scan_data[WORKGROUP_SIZE];
// Stores the base offset for the entire workgroup in the global output buffer
shared uint s_workgroupBaseOffset;
// Stores the total active count for the entire workgroup (read during scan)
shared uint s_workgroupTotalActiveCount;


void main() {
    // --- Basic Setup ---
    uint tid = gl_LocalInvocationIndex; // Thread index within workgroup (0 to WORKGROUP_SIZE-1)
    uint globalBlockID1D = gl_GlobalInvocationID.x; // Global 1D block index

    uvec3 blockGridDim = pc.blockGridDim.xyz;
    uint totalBlocks = blockGridDim.x * blockGridDim.y * blockGridDim.z;

    // Default values in case of early exit or inactivity
    uint activeFlag = 0;
    uvec2 minMax = uvec2(0, 0);

    // --- Bounds Check & Activity Check ---
    // Only process if within the valid range of blocks
    if (globalBlockID1D < totalBlocks) {
        // Calculate 3D block coordinates
        uvec3 blockCoord;
        uint planeSize = blockGridDim.x * blockGridDim.y;
        blockCoord.z = (planeSize > 0) ? (globalBlockID1D / planeSize) : 0;
        uint remainder = (planeSize > 0) ? (globalBlockID1D % planeSize) : globalBlockID1D;
        blockCoord.y = (blockGridDim.x > 0) ? (remainder / blockGridDim.x) : 0;
        blockCoord.x = (blockGridDim.x > 0) ? (remainder % pc.blockGridDim.x) : remainder;

        // Read min/max for this block
        minMax = imageLoad(minMaxInputVolume, ivec3(blockCoord)).xy;

        // Determine activity using integer comparison
        bool blockIsActive = false;
        if (minMax.x != minMax.y) {
            blockIsActive = (float(pc.isovalue) >= minMax.x && float(pc.isovalue) <= minMax.y);
        }
        activeFlag = blockIsActive ? 1 : 0;
    }
    // Threads outside totalBlocks range will have activeFlag = 0

    // --- Shared Memory Parallel Exclusive Scan ---

    // Stage 1: Load input (activeFlag) into shared memory
    s_scan_data[tid] = activeFlag;
    barrier(); // Ensure all loads complete before starting scan

    // Stage 2: Up-Sweep (Reduction Phase) - builds sum tree in shared memory
    // In each step, threads add values from power-of-2 strides away.
    for (uint stride = 1; stride < WORKGROUP_SIZE; stride *= 2) {
        uint index = (tid + 1) * stride * 2 - 1;
        if (index < WORKGROUP_SIZE) {
            s_scan_data[index] += s_scan_data[index - stride];
        }
        barrier(); // Sync after each step
    }

    // Stage 3: Store total sum, clear last element for exclusive scan
    // Thread 0 is responsible for handling the total sum and preparing for down-sweep
    if (tid == 0) {
        // The last element now holds the total sum for the workgroup
        s_workgroupTotalActiveCount = s_scan_data[WORKGROUP_SIZE - 1]; // Store total count
        s_scan_data[WORKGROUP_SIZE - 1] = 0; // Clear last element for exclusive scan down-sweep
    }
    // Synchronize: Ensure total is stored by thread 0 and last element is cleared
    // before any thread starts the down-sweep. Also makes total count visible.
    barrier();

    // *** DEBUG FOCUS AREA 1: Down-Sweep Logic ***
    // Stage 4: Down-Sweep (Scan Phase) - builds exclusive scan result in s_scan_data
    for (uint stride = WORKGROUP_SIZE / 2; stride > 0; stride /= 2) {
        uint index = (tid + 1) * stride * 2 - 1;
        if (index < WORKGROUP_SIZE) {
            uint temp = s_scan_data[index - stride];    // Cache 'left' value
            s_scan_data[index - stride] = s_scan_data[index]; // Write 'right' value to 'left' position
            s_scan_data[index] = temp + s_scan_data[index]; // Write sum ('left' + 'right') to 'right' position
        }
        // *** Add Debug Breakpoint Here: Inspect s_scan_data evolution ***
        // Does the array content match the expected state after each step of the down-sweep?
        barrier(); // Sync after each step
    }

    // Result of scan: s_scan_data[tid] now holds the exclusive prefix sum for thread 'tid'
    uint localOffset = s_scan_data[tid];
    // *** Add Debug Breakpoint Here: Inspect final localOffset for different 'tid' ***
    // Is it the correct exclusive prefix sum? Is it consistent across runs?

    // --- Global Offset Calculation ---
    // Stage 5 & 6: Thread 0 gets global base offset via atomic add and broadcasts
    if (tid == 0) {
        uint wgTotal = s_workgroupTotalActiveCount; // Read workgroup total calculated earlier
        s_workgroupBaseOffset = atomicAdd(activeBlockCount.count, wgTotal);
        // *** Add Debug Breakpoint Here (for tid==0): Inspect wgTotal, s_workgroupBaseOffset ***
    }
    // *** DEBUG FOCUS AREA 2: Barrier Sufficiency & Broadcast Read ***
    // Synchronize: Ensure base offset is written by thread 0 and visible to all threads
    barrier(); // Is this barrier sufficient to guarantee visibility of s_workgroupBaseOffset?

    // All threads read the workgroup base offset
    uint workgroupBaseOffset = s_workgroupBaseOffset;
    // *** Add Debug Breakpoint Here: Inspect workgroupBaseOffset for different 'tid'. Is it consistent? ***

    // --- Final Write ---
    // Stage 7: Calculate final global offset
    uint globalOffset = workgroupBaseOffset + localOffset;
    // *** Add Debug Breakpoint Here: Inspect final globalOffset for active threads ***
    // Are the offsets unique and ordered correctly?

    // Stage 8: Active threads write their original ID to the calculated offset
    // Ensure the original activeFlag (from before scan) is used here
    if (activeFlag == 1) {
        // Ensure globalOffset is within the bounds of the output buffer
        // (Assuming CompactedBlockIDs buffer is sized for totalBlocks worst case)
        if (globalOffset < totalBlocks) { // Basic bounds check
            blockIDs[globalOffset] = globalBlockID1D;
        }
        // else { /* Handle error or out-of-bounds write if necessary */ }
    }
}
