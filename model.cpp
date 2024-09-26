#include <stdint.h>
#include <exception>
#include <fstream>
#include <iostream>
#include <json.hpp>

using json = nlohmann::json;

int main(int argc, char** argv) {
    if (argc < 3) {
        throw std::runtime_error(
            "Not enough arguments."
            " Usage: ./model path/to/config.json path/to/model.safetensors");
    }

    {
        std::ifstream in(argv[1]);
        auto config = json::parse(in);
        std::cerr << config["num_hidden_layers"] << std::endl;
    }

    {
        std::ifstream in(argv[2]);
        uint64_t nHeader(0);
        in.read(reinterpret_cast<char*>(&nHeader), sizeof(nHeader));
        std::string headerData(nHeader, '\0');
        in.read(headerData.data(), headerData.size());
        auto header = json::parse(headerData);

        std::cerr << nHeader << " header bytes" << std::endl;
        std::cerr << header["model.layers.0.self_attn.k_proj.weight"] << std::endl;
    }

    return 0;
}
