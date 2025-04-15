#pragma once

struct Volume {
    glm::uvec3 volume_dims;
    std::string volume_type;
    std::vector<uint8_t> volume_data;
};

struct alignas(16) PushConstants {
    glm::uvec4 volumeDim;    // Offset 0, Size 16. Shader uses .xyz
    glm::uvec4 blockDim;     // Offset 16, Size 16. Shader uses .xyz
    glm::uvec4 blockGridDim; // Offset 32, Size 16. Shader uses .xyz
    float      isovalue;     // Offset 48, Size 4.
};


inline Volume loadVolume(char **argv, const char *path)
{
    std::string spath = argv[0];
    std::string::size_type pos = spath.find_last_of("/\\");
    if (pos == std::string::npos)
        spath = "";
    else
        spath = spath.substr(0, pos + 1);
    spath += path;

    Volume volume = {};
    const std::string file = path;
    //const float isovalue = std::stof(argv[2]);
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

    const size_t volume_bytes =
        static_cast<size_t>(volume_dims.x) * static_cast<size_t>(volume_dims.y) * static_cast<size_t>(volume_dims.z);
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
    return volume;
}