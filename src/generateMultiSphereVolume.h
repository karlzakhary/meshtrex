#pragma once

#include <vector>
#include <cstdint>
#include <glm/glm.hpp>
#include <random>
#include <cmath>
#include <algorithm>

// Generate a volume with multiple spheres for testing occlusion culling
class MultiSphereVolumeGenerator {
public:
    struct Sphere {
        glm::vec3 center;
        float radius;
        uint8_t density;  // Density value for this sphere
    };
    
    // Generate volume with many small spheres
    static std::vector<uint8_t> generateMultiSphereVolume(
        int width, int height, int depth,
        int numSpheres = 1000,
        float minRadius = 2.0f,
        float maxRadius = 8.0f,
        bool randomPlacement = true,
        bool allowOverlap = true) {
        
        std::vector<uint8_t> data(width * height * depth, 0);
        std::vector<Sphere> spheres;
        
        // Random number generator
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> radiusDist(minRadius, maxRadius);
        std::uniform_real_distribution<float> xDist(maxRadius, width - maxRadius);
        std::uniform_real_distribution<float> yDist(maxRadius, height - maxRadius);
        std::uniform_real_distribution<float> zDist(maxRadius, depth - maxRadius);
        std::uniform_int_distribution<int> densityDist(100, 255);
        
        // Generate sphere positions
        for (int i = 0; i < numSpheres; i++) {
            Sphere sphere;
            
            if (randomPlacement) {
                sphere.center.x = xDist(gen);
                sphere.center.y = yDist(gen);
                sphere.center.z = zDist(gen);
                sphere.radius = radiusDist(gen);
            } else {
                // Grid placement for systematic testing
                int gridSize = std::cbrt(numSpheres) + 1;
                float spacing = std::min({width, height, depth}) / float(gridSize);
                int gridX = i % gridSize;
                int gridY = (i / gridSize) % gridSize;
                int gridZ = i / (gridSize * gridSize);
                
                sphere.center.x = spacing * (gridX + 0.5f);
                sphere.center.y = spacing * (gridY + 0.5f);
                sphere.center.z = spacing * (gridZ + 0.5f);
                sphere.radius = minRadius + (maxRadius - minRadius) * (i % 3) / 2.0f;
            }
            
            sphere.density = densityDist(gen);
            
            // Check for overlap if needed
            if (!allowOverlap && !spheres.empty()) {
                bool overlaps = false;
                for (const auto& other : spheres) {
                    float dist = glm::distance(sphere.center, other.center);
                    if (dist < sphere.radius + other.radius) {
                        overlaps = true;
                        break;
                    }
                }
                if (overlaps) {
                    i--;  // Retry this sphere
                    continue;
                }
            }
            
            spheres.push_back(sphere);
        }
        
        // Rasterize spheres into volume
        for (const auto& sphere : spheres) {
            int minX = std::max(0, int(sphere.center.x - sphere.radius - 1));
            int maxX = std::min(width - 1, int(sphere.center.x + sphere.radius + 1));
            int minY = std::max(0, int(sphere.center.y - sphere.radius - 1));
            int maxY = std::min(height - 1, int(sphere.center.y + sphere.radius + 1));
            int minZ = std::max(0, int(sphere.center.z - sphere.radius - 1));
            int maxZ = std::min(depth - 1, int(sphere.center.z + sphere.radius + 1));
            
            for (int z = minZ; z <= maxZ; z++) {
                for (int y = minY; y <= maxY; y++) {
                    for (int x = minX; x <= maxX; x++) {
                        float dist = std::sqrt(
                            (x - sphere.center.x) * (x - sphere.center.x) +
                            (y - sphere.center.y) * (y - sphere.center.y) +
                            (z - sphere.center.z) * (z - sphere.center.z)
                        );
                        
                        if (dist <= sphere.radius) {
                            int idx = x + y * width + z * width * height;
                            // Use max to handle overlapping spheres
                            data[idx] = std::max(data[idx], sphere.density);
                        }
                    }
                }
            }
        }
        
        return data;
    }
    
    // Generate volume with spheres arranged in layers (good for occlusion testing)
    static std::vector<uint8_t> generateLayeredSphereVolume(
        int width, int height, int depth,
        int spheresPerLayer = 100,
        int numLayers = 10) {
        
        std::vector<uint8_t> data(width * height * depth, 0);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        
        float layerSpacing = depth / float(numLayers);
        
        for (int layer = 0; layer < numLayers; layer++) {
            float z = layerSpacing * (layer + 0.5f);
            
            // Vary sphere size by layer (front layers have larger spheres for better occlusion)
            float radius = 12.0f + layer * 1.5f;  // Increased from 3.0f + layer * 0.5f
            std::uniform_real_distribution<float> xDist(radius, width - radius);
            std::uniform_real_distribution<float> yDist(radius, height - radius);
            std::uniform_int_distribution<int> densityDist(100 + layer * 10, 255);
            
            for (int i = 0; i < spheresPerLayer; i++) {
                float x = xDist(gen);
                float y = yDist(gen);
                uint8_t density = densityDist(gen);
                
                // Rasterize sphere
                int minX = std::max(0, int(x - radius - 1));
                int maxX = std::min(width - 1, int(x + radius + 1));
                int minY = std::max(0, int(y - radius - 1));
                int maxY = std::min(height - 1, int(y + radius + 1));
                int minZ = std::max(0, int(z - radius - 1));
                int maxZ = std::min(depth - 1, int(z + radius + 1));
                
                for (int vz = minZ; vz <= maxZ; vz++) {
                    for (int vy = minY; vy <= maxY; vy++) {
                        for (int vx = minX; vx <= maxX; vx++) {
                            float dist = std::sqrt(
                                (vx - x) * (vx - x) +
                                (vy - y) * (vy - y) +
                                (vz - z) * (vz - z)
                            );
                            
                            if (dist <= radius) {
                                int idx = vx + vy * width + vz * width * height;
                                data[idx] = std::max(data[idx], density);
                            }
                        }
                    }
                }
            }
        }
        
        return data;
    }
    
    // Generate a stress test with dense spheres to test occlusion
    static std::vector<uint8_t> generateOcclusionStressTest(
        int width, int height, int depth) {
        
        // Create a wall of spheres in front, then scattered spheres behind
        std::vector<uint8_t> data(width * height * depth, 0);
        
        // Front wall of large spheres (should occlude many behind)
        int wallDepth = depth / 4;
        int spheresInWall = 500;
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> wallX(15.0f, width - 15.0f);
        std::uniform_real_distribution<float> wallY(15.0f, height - 15.0f);
        std::uniform_real_distribution<float> wallZ(5.0f, wallDepth);
        
        // Create front wall with larger spheres
        for (int i = 0; i < spheresInWall; i++) {
            float x = wallX(gen);
            float y = wallY(gen);
            float z = wallZ(gen);
            float radius = 12.0f;  // Increased from 4.0f
            
            drawSphere(data, width, height, depth, x, y, z, radius, 200);
        }
        
        // Create scattered spheres behind the wall
        std::uniform_real_distribution<float> backX(15.0f, width - 15.0f);
        std::uniform_real_distribution<float> backY(15.0f, height - 15.0f);
        std::uniform_real_distribution<float> backZ(wallDepth + 20.0f, depth - 15.0f);
        
        for (int i = 0; i < 1000; i++) {
            float x = backX(gen);
            float y = backY(gen);
            float z = backZ(gen);
            float radius = 10.0f + (gen() % 10);  // Increased from 3.0f + (gen() % 5)
            
            drawSphere(data, width, height, depth, x, y, z, radius, 150);
        }
        
        return data;
    }
    
private:
    static void drawSphere(std::vector<uint8_t>& data, int width, int height, int depth,
                          float cx, float cy, float cz, float radius, uint8_t density) {
        int minX = std::max(0, int(cx - radius - 1));
        int maxX = std::min(width - 1, int(cx + radius + 1));
        int minY = std::max(0, int(cy - radius - 1));
        int maxY = std::min(height - 1, int(cy + radius + 1));
        int minZ = std::max(0, int(cz - radius - 1));
        int maxZ = std::min(depth - 1, int(cz + radius + 1));
        
        for (int z = minZ; z <= maxZ; z++) {
            for (int y = minY; y <= maxY; y++) {
                for (int x = minX; x <= maxX; x++) {
                    float dist = std::sqrt(
                        (x - cx) * (x - cx) +
                        (y - cy) * (y - cy) +
                        (z - cz) * (z - cz)
                    );
                    
                    if (dist <= radius) {
                        int idx = x + y * width + z * width * height;
                        data[idx] = std::max(data[idx], density);
                    }
                }
            }
        }
    }
};