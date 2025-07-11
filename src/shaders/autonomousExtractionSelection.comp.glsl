#version 460 core
#extension GL_EXT_scalar_block_layout : enable

// Autonomous Extraction Selection Shader
// Decides which pages need extraction based on:
// PRIMARY: Isovalue changes
// SECONDARY: Camera frustum (only if volume doesn't fit in memory)

#define WORKGROUP_SIZE 64

// Command types
#define EXTRACT_CMD_FULL         0  // Extract entire page
#define EXTRACT_CMD_CONDITIONAL  1  // Extract if meets criteria
#define EXTRACT_CMD_SKIP         2  // Skip extraction
#define EXTRACT_CMD_EVICT        3  // Evict page

layout(local_size_x = WORKGROUP_SIZE) in;

// Extraction state
layout(set = 0, binding = 0, std430) buffer ExtractionState {
    float currentParameter;      // Current isovalue
    float previousParameter;     // Previous isovalue
    uint framesSinceChange;
    uint parameterChanged;       // 1 if changed
    uint totalPagesToExtract;
    uint pagesExtractedSoFar;
    uint extractionComplete;
    uint padding;
} extractionState;

// Memory state
layout(set = 0, binding = 1, std430) readonly buffer MemoryState {
    uint totalPages;
    uint maxResidentPages;
    uint currentResidentPages;
    uint entireVolumeFits;      // 1 if volume fits in memory
    uint memoryPressure;        // 0-100
    uint visiblePageCount;
    uint padding[2];
} memoryState;

// Page validity (1 = valid for current isovalue, 0 = needs extraction)
layout(set = 0, binding = 2, std430) buffer PageValidity {
    uint validity[];
} pageValidity;

// Command queue
layout(set = 0, binding = 3, std430) buffer CommandQueue {
    uint commandCount;
    uint commands[];  // AutonomousCommand structs
} commandQueue;

// Page residency
layout(set = 0, binding = 4, std430) readonly buffer PageResidency {
    uint residency[];  // 1 = resident, 0 = not resident
} pageResidency;

// Camera data (only used if volume doesn't fit)
layout(set = 0, binding = 5, std140) uniform CameraData {
    mat4 viewMatrix;
    mat4 projMatrix;
    vec4 frustumPlanes[6];
} camera;

// Page priority
layout(set = 0, binding = 6, std430) buffer PagePriority {
    float priority[];
} pagePriority;

// Push constants
layout(push_constant) uniform PushConstants {
    uint volumeDimX;
    uint volumeDimY;
    uint volumeDimZ;
    uint pageSizeX;
    uint pageSizeY;
    uint pageSizeZ;
    uint maxCommandsPerFrame;
    uint currentPass;  // 0=analyze, 1=prioritize, 2=generate
} pc;

// Helper functions
uvec3 getPageCoord(uint pageIndex) {
    uint pagesX = (pc.volumeDimX + pc.pageSizeX - 1) / pc.pageSizeX;
    uint pagesY = (pc.volumeDimY + pc.pageSizeY - 1) / pc.pageSizeY;
    
    uint z = pageIndex / (pagesX * pagesY);
    uint rem = pageIndex % (pagesX * pagesY);
    uint y = rem / pagesX;
    uint x = rem % pagesX;
    
    return uvec3(x, y, z);
}

vec3 getPageCenter(uvec3 pageCoord) {
    return vec3(pageCoord.x * pc.pageSizeX + pc.pageSizeX / 2,
                pageCoord.y * pc.pageSizeY + pc.pageSizeY / 2,
                pageCoord.z * pc.pageSizeZ + pc.pageSizeZ / 2);
}

bool isPageInFrustum(vec3 pageCenter) {
    // Calculate page radius based on the diagonal of the non-uniform page
    float pageDiagonal = sqrt(float(pc.pageSizeX * pc.pageSizeX + 
                                   pc.pageSizeY * pc.pageSizeY + 
                                   pc.pageSizeZ * pc.pageSizeZ));
    float pageRadius = pageDiagonal * 0.5;
    
    for (int i = 0; i < 6; i++) {
        float distance = dot(vec4(pageCenter, 1.0), camera.frustumPlanes[i]);
        if (distance < -pageRadius) {
            return false;
        }
    }
    return true;
}

void main() {
    uint pageIndex = gl_GlobalInvocationID.x;
    if (pageIndex >= memoryState.totalPages) return;
    
    if (pc.currentPass == 0) {
        // Pass 0: Analyze extraction state
        
        // Check if isovalue changed
        if (pageIndex == 0 && extractionState.parameterChanged != 0) {
            // Mark ALL pages as invalid when isovalue changes
            extractionState.totalPagesToExtract = memoryState.totalPages;
            extractionState.pagesExtractedSoFar = 0;
            extractionState.extractionComplete = 0;
        }
        
        // If isovalue changed, invalidate this page
        if (extractionState.parameterChanged != 0) {
            pageValidity.validity[pageIndex] = 0;  // Needs extraction
            pagePriority.priority[pageIndex] = 1.0; // High priority
        }
    }
    else if (pc.currentPass == 1) {
        // Pass 1: Prioritize pages
        
        uint isResident = pageResidency.residency[pageIndex];
        uint isValid = pageValidity.validity[pageIndex];
        
        if (isResident == 0 || isValid != 0) {
            // Page not resident or already valid
            pagePriority.priority[pageIndex] = 0.0;
            return;
        }
        
        // Default priority for extraction
        float priority = 1.0;
        
        // If volume doesn't fit in memory, use camera frustum for prioritization
        if (memoryState.entireVolumeFits == 0) {
            uvec3 pageCoord = getPageCoord(pageIndex);
            vec3 pageCenter = getPageCenter(pageCoord);
            
            if (isPageInFrustum(pageCenter)) {
                priority = 2.0;  // Higher priority for visible pages
            } else {
                priority = 0.5;  // Lower priority for non-visible pages
            }
        }
        
        pagePriority.priority[pageIndex] = priority;
    }
    else if (pc.currentPass == 2) {
        // Pass 2: Generate extraction commands
        
        uint isResident = pageResidency.residency[pageIndex];
        uint isValid = pageValidity.validity[pageIndex];
        float priority = pagePriority.priority[pageIndex];
        
        // Skip if page doesn't need extraction
        if (isResident == 0 || isValid != 0 || priority == 0.0) {
            return;
        }
        
        // Try to allocate command slot
        uint cmdIndex = atomicAdd(commandQueue.commandCount, 1);
        if (cmdIndex >= pc.maxCommandsPerFrame) {
            atomicAdd(commandQueue.commandCount, -1);  // Rollback
            return;
        }
        
        // Generate extraction command
        uvec3 pageCoord = getPageCoord(pageIndex);
        
        // Pack command (8 uints per command)
        uint baseIndex = cmdIndex * 8;
        commandQueue.commands[baseIndex + 0] = EXTRACT_CMD_FULL;
        commandQueue.commands[baseIndex + 1] = pageCoord.x;
        commandQueue.commands[baseIndex + 2] = pageCoord.y;
        commandQueue.commands[baseIndex + 3] = pageCoord.z;
        commandQueue.commands[baseIndex + 4] = 0;  // mipLevel
        commandQueue.commands[baseIndex + 5] = floatBitsToUint(priority);
        commandQueue.commands[baseIndex + 6] = 1;  // requiresExtraction
        commandQueue.commands[baseIndex + 7] = 0;  // padding
        
        // Mark page as being processed
        pageValidity.validity[pageIndex] = 1;
        atomicAdd(extractionState.pagesExtractedSoFar, 1);
        
        // Check if extraction is complete
        if (extractionState.pagesExtractedSoFar >= extractionState.totalPagesToExtract) {
            extractionState.extractionComplete = 1;
            extractionState.parameterChanged = 0;  // Reset change flag
        }
    }
}

// This shader autonomously:
// 1. Detects isovalue changes and marks ALL pages for re-extraction
// 2. Prioritizes pages based on:
//    - If volume fits in memory: Extract everything
//    - If volume too large: Prioritize visible pages
// 3. Generates extraction commands up to maxCommandsPerFrame
// 4. Tracks extraction progress