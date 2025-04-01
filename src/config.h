// Should we enable vertical sync during presentation? Worth setting to 0 when doing perf profiling to avoid GPU downclock during idle
#define CONFIG_VSYNC 1

// Should we enable synchronization validation? Worth running with 1 occasionally to check correctness.
#define CONFIG_SYNCVAL 0

// Maximum number of texture descriptors in the pool
#define DESCRIPTOR_LIMIT 65536
