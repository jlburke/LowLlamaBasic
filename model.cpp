#include <stdint.h>
#include <exception>
#include <fstream>
#include <iostream>
#define JSON_USE_IMPLICIT_CONVERSIONS 0
#include <json.hpp>

using json = nlohmann::json;

namespace lp {

struct parameter {
    std::vector<unsigned> shape;
    const void* data;
};

struct model {
    parameter embed_tokens;
};

parameter loadParameter(const json& j, const char* data) {
    if (j["dtype"].template get<std::string>() != "BF16") {
        throw std::invalid_argument("Non-BF16 data");
    }
    auto start = j["data_offsets"][0].template get<uint64_t>();
    return {j["shape"].template get<std::vector<unsigned>>(), data + start};
}

float bf16_to_float(int16_t value) {
    union {
        float f;
        int16_t i[2];
    } u;
    u.i[0] = 0;
    u.i[1] = value;
    return u.f;
}

}  // namespace lp

int main(int argc, char** argv) {
    if (argc < 3) {
        throw std::runtime_error(
            "Not enough arguments."
            " Usage: ./model path/to/config.json path/to/model.safetensors");
    }

    {
        std::ifstream in(argv[1]);
        auto config = json::parse(in);
        std::cerr << config["num_hidden_layers"] << " hidden layers" << std::endl;
    }

    {
        std::ifstream in(argv[2]);
        uint64_t nHeader(0);
        in.read(reinterpret_cast<char*>(&nHeader), sizeof(nHeader));
        std::string headerData(nHeader, '\0');
        in.read(headerData.data(), headerData.size());
        auto header = json::parse(headerData);
        header.erase("__metadata__");
        uint64_t maxOffset(0);
        for (auto e : header.items()) {
            maxOffset = std::max(maxOffset, e.value()["data_offsets"][1].template get<uint64_t>());
        }

        std::vector<char> tensorData;
        tensorData.reserve(maxOffset);
        constexpr uint64_t chunkSize(1 << 20);
        for (auto i = uint64_t(0); i < maxOffset; i += chunkSize) {
            in.read(tensorData.data() + i, std::min(chunkSize, maxOffset - i));
        }

        auto embedTokens =
            lp::loadParameter(header["model.embed_tokens.weight"], tensorData.data());

        auto p = reinterpret_cast<const int16_t*>(embedTokens.data);
        std::cerr << "embed_tokens[0,0] = " << lp::bf16_to_float(p[0]) << std::endl;
        std::cerr << "embed_tokens[0,1] = " << lp::bf16_to_float(p[1]) << std::endl;
    }

    return 0;
}
