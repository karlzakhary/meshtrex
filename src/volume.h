#pragma once

#include <regex>
#include <iostream>
#include <fstream>


struct Volume {
    glm::uvec3 volume_dims;
    std::string volume_type;
    std::vector<uint8_t> volume_data;
    size_t voxel_size = 0;
};

inline Volume loadVolume(const char *path)
{
    Volume volume = {};
    const std::string file = path;
    const std::regex match_filename(R"((\w+)_(\d+)x(\d+)x(\d+)_(.+)\.raw)");
    auto matches = std::sregex_iterator(file.begin(), file.end(), match_filename);
    if (matches == std::sregex_iterator() || matches->size() != 6) {
        std::cerr << "Unrecognized raw volume naming scheme, expected a format like: "
                  << "'<name>_<X>x<Y>x<Z>_<data type>.raw' but '" << file << "' did not match"
                  << std::endl;
        throw std::runtime_error("Invalaid raw file naming scheme");
    }
    const glm::uvec3 volume_dims(
        std::stoi((*matches)[2]), std::stoi((*matches)[3]), std::stoi((*matches)[4]));
    const std::string volume_type = (*matches)[5];
    size_t voxel_size = 0;

    if (volume_type == "uint8") {
        voxel_size = 1;
    } else if (volume_type == "uint16") {
        voxel_size = 2;
    } else if (volume_type == "float32") {
        voxel_size = 4;
    } else {
        std::cout << "Volume type '" << volume_type
                  << "' support was not built, please recompile" << std::endl;
        throw std::runtime_error("Rebuild with " + volume_type + " support");
    }
    
    const size_t volume_bytes =
        static_cast<size_t>(volume_dims.x) * static_cast<size_t>(volume_dims.y) * static_cast<size_t>(volume_dims.z) * voxel_size;
    std::vector<uint8_t> volume_data(volume_bytes, 0);
    std::ifstream fin(file.c_str(), std::ios::binary);
    if (!fin) {
        std::cerr << "Failed to open " << file << "\n";
    }
    if (!fin.read(reinterpret_cast<char *>(volume_data.data()), volume_bytes)) {
        std::cerr << "Failed to read volume data\n";
    }
    volume.volume_dims = volume_dims;
    volume.volume_type = volume_type;
    volume.volume_data = volume_data;
    volume.voxel_size = voxel_size;
    return volume;
}