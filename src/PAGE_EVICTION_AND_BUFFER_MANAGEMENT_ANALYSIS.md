# Page Eviction and Buffer Management Analysis

## Page/Brick Eviction Mechanics

### Current Eviction Triggers

#### 1. **Time-Based Eviction (LRU - Least Recently Used)**
```cpp
// From VolumeStreamer::updateStreaming()
for (auto& [coord, entry] : pageMap) {
    if (entry.isResident && (frameIndex - entry.lastUsedFrame) > 60) {
        pagesToEvict.push_back(coord);
    }
}
```

**Trigger**: Pages unused for more than 60 frames are evicted
**Policy**: LRU (Least Recently Used)
**Frequency**: Every frame during `beginFrame()`

#### 2. **Memory Pressure Eviction**
```cpp
// From VolumeStreamer::loadPage()
if (freeAtlasPages.empty()) {
    return; // Cannot load new page - no free atlas slots
}
```

**Trigger**: When atlas is full and new pages are requested
**Current Issue**: ❌ **No forced eviction implemented** - new page requests fail

#### 3. **Manual Eviction**
```cpp
void VolumeStreamer::evictPage(const PageCoord& coord) {
    // Remove from pageMap and add atlas slot back to freeAtlasPages
}
```

**Trigger**: Called manually by application or during time-based cleanup

### Eviction Process Workflow

```
Frame N: Page Requested
    ↓
Check Atlas Capacity
    ├─ Has Free Slots → Load Page
    └─ Atlas Full → ❌ Request Fails (ISSUE!)
    
Frame N+60: Time-Based Cleanup
    ↓
Check lastUsedFrame for all pages
    ↓
Evict old pages
    ↓ 
Update Page Table (mark as non-resident)
    ↓
Add atlas slots back to free pool
```

## Critical Issues with Current Eviction

### ❌ **Issue 1: No Forced Eviction Under Memory Pressure**

**Problem**: When atlas is full, new page requests simply fail rather than evicting old pages.

**Code Location**: `VolumeStreamer::loadPage()` lines 257-259
```cpp
if (freeAtlasPages.empty()) {
    return; // This should trigger eviction instead!
}
```

**Impact**: 
- GPU may generate commands for pages that can't be loaded
- Streaming becomes non-responsive under high memory pressure
- No adaptive behavior when working set exceeds atlas capacity

### ❌ **Issue 2: Race Condition Between GPU Command Generation and Eviction**

**Problem**: GPU generates commands based on current page table, but pages can be evicted between command generation and execution.

**Timeline**:
```
T1: GPU Command Generation reads page table → generates commands for Page X
T2: CPU evicts Page X due to memory pressure
T3: GPU executes commands for Page X → accesses non-resident page
```

**Impact**: GPU may access invalid atlas regions or get undefined data

## Global Buffer Overflow in Extraction Pass

### Current Buffer Sizing Strategy

The extraction pass creates **per-page** buffers sized based on active blocks:

```cpp
// Per-page buffer sizing
const VkDeviceSize MAX_TOTAL_VERTICES_BYTES =
    static_cast<VkDeviceSize>(filteringOutput.activeBlockCount) *
    cellsPerBlock * MAX_VERTS_PER_CELL * sizeof(StreamingVertexData);
```

### ❌ **Critical Problem: No Global Buffer Management**

**Issue**: Each page creates its own extraction buffers, leading to:

1. **Memory Fragmentation**: Hundreds of small buffers instead of efficient global pools
2. **No Cross-Page Coherency**: Can't render meshlets from multiple pages together
3. **GPU Memory Explosion**: Memory usage = O(resident_pages * max_geometry_per_page)

### Potential Overflow Scenarios

#### Scenario 1: High-Detail Volume
```
Volume: 1024³ with 32³ pages = 32,768 total pages
Atlas Capacity: 1,024 resident pages
Geometry per page: ~1M vertices, ~3M indices, ~10K meshlets

Worst Case Memory:
- Vertices: 1,024 pages × 1M vertices × 48 bytes = 48 GB ❌
- Indices: 1,024 pages × 3M indices × 4 bytes = 12 GB ❌  
- Meshlets: 1,024 pages × 10K meshlets × 16 bytes = 160 MB
```

#### Scenario 2: Rapid Page Turnover
```
Frame Rate: 60 FPS
Page Eviction: Every 60 frames (1 second)
Page Turnover: 1,024 pages/second

Memory Allocation Rate: 60 GB/second ❌
GPU Memory Exhaustion: Inevitable
```

## Proposed Solutions

### 1. **Implement Forced Eviction Under Memory Pressure**

```cpp
// Revised VolumeStreamer::loadPage()
void VolumeStreamer::loadPage(const PageCoord& coord) {
    uint32_t atlasPageIndex = allocateAtlasPage();
    
    if (atlasPageIndex == UINT32_MAX) {
        // Force eviction of least recently used page
        PageCoord evictCoord = findLRUPage();
        evictPage(evictCoord);
        atlasPageIndex = allocateAtlasPage();
        
        if (atlasPageIndex == UINT32_MAX) {
            throw std::runtime_error("Critical: Cannot evict pages for new allocation");
        }
    }
    
    // Continue with page loading...
}

PageCoord VolumeStreamer::findLRUPage() {
    PageCoord lruCoord;
    uint32_t oldestFrame = UINT32_MAX;
    
    for (const auto& [coord, entry] : pageMap) {
        if (entry.isResident && entry.lastUsedFrame < oldestFrame) {
            oldestFrame = entry.lastUsedFrame;
            lruCoord = coord;
        }
    }
    return lruCoord;
}
```

### 2. **Implement Global Buffer Pools for Extraction**

```cpp
// Global extraction buffer manager
class GlobalExtractionBuffers {
public:
    struct BufferPool {
        Buffer globalVertexBuffer;      // e.g., 100M vertices total
        Buffer globalIndexBuffer;       // e.g., 300M indices total  
        Buffer globalMeshletBuffer;     // e.g., 1M meshlets total
        
        // Atomic allocation counters
        Buffer vertexAllocCounter;
        Buffer indexAllocCounter;
        Buffer meshletAllocCounter;
        
        // High watermark tracking
        uint32_t maxVerticesUsed = 0;
        uint32_t maxIndicesUsed = 0;
        uint32_t maxMeshletsUsed = 0;
    };
    
    // Allocate from global pools instead of per-page buffers
    AllocationResult allocateForPage(const PageCoord& coord, 
                                   uint32_t estimatedVertices,
                                   uint32_t estimatedIndices,
                                   uint32_t estimatedMeshlets);
                                   
    // Reset pools each frame
    void resetAllocations();
    
    // Handle overflow gracefully
    void handleOverflow(BufferType type);
};
```

### 3. **Implement GPU-Side Overflow Protection**

```glsl
// In streaming extraction shaders
layout(set = 0, binding = N) buffer GlobalAllocationCounters {
    uint globalVertexCount;
    uint globalIndexCount;
    uint globalMeshletCount;
    
    uint maxVertices;
    uint maxIndices;
    uint maxMeshlets;
    
    uint overflowFlags; // Bit flags for overflow detection
};

// Safe allocation in shader
uint allocateVertices(uint count) {
    uint offset = atomicAdd(globalVertexCount, count);
    
    if (offset + count > maxVertices) {
        atomicOr(overflowFlags, VERTEX_OVERFLOW_FLAG);
        return INVALID_OFFSET; // Graceful degradation
    }
    
    return offset;
}
```

### 4. **Implement Frame-Coherent Eviction**

```cpp
// Synchronize eviction with GPU command execution
class FrameCoherentEviction {
    struct FrameData {
        std::vector<PageCoord> pagesUsedThisFrame;
        VkFence frameFence;
        uint32_t frameIndex;
    };
    
    std::queue<FrameData> frameHistory;
    
public:
    void markPageUsed(const PageCoord& coord, uint32_t frameIndex) {
        currentFrame.pagesUsedThisFrame.push_back(coord);
    }
    
    void endFrame(VkCommandBuffer cmd, uint32_t frameIndex) {
        // Insert fence after GPU work
        vkCmdSetEvent(cmd, frameCompletionEvent, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
        
        currentFrame.frameIndex = frameIndex;
        frameHistory.push(std::move(currentFrame));
        currentFrame = {};
    }
    
    void safeEvictPages() {
        // Only evict pages from frames that have completed GPU execution
        while (!frameHistory.empty()) {
            auto& frame = frameHistory.front();
            
            if (vkGetFenceStatus(device, frame.frameFence) == VK_SUCCESS) {
                // Safe to evict pages from this frame
                for (const auto& coord : frame.pagesUsedThisFrame) {
                    // Update last used frame and mark eligible for eviction
                }
                frameHistory.pop();
            } else {
                break; // This frame hasn't completed yet
            }
        }
    }
};
```

## Recommended Implementation Priority

### Priority 1: **Critical Buffer Overflow Prevention**
1. ✅ Implement global buffer pools
2. ✅ Add GPU-side overflow detection
3. ✅ Graceful degradation when buffers full

### Priority 2: **Robust Eviction Policy**  
1. ✅ Add forced eviction under memory pressure
2. ✅ Implement frame-coherent eviction
3. ✅ Better LRU tracking with access patterns

### Priority 3: **Performance Optimization**
1. ✅ Predictive page loading based on camera movement
2. ✅ Hierarchical eviction (evict lower LOD pages first)
3. ✅ Multi-threaded page loading/eviction

## Current Status: ⚠️ **Critical Issues Need Addressing**

The current implementation has significant robustness issues that will cause failures under realistic workloads:

- **Memory Pressure**: No forced eviction when atlas is full
- **Buffer Overflow**: No protection against extraction buffer exhaustion  
- **Race Conditions**: GPU/CPU synchronization issues during eviction
- **Memory Efficiency**: Per-page buffers cause massive memory waste

These issues must be addressed before the streaming system can handle production workloads.