#include <stdint.h>
#include <cmath>
#include <exception>
#include <format>
#include <fstream>
#include <iostream>
#include <vector>

#define JSON_USE_IMPLICIT_CONVERSIONS 0
#include <json.hpp>

using json = nlohmann::json;

namespace lp {

///////////////////////////////////////////////////////////////////////////////
// Basics

using bf16 = int16_t;

float bf16_to_float(bf16 value) {
    union {
        float f;
        int16_t i[2];
    } u;
    u.i[0] = 0;
    u.i[1] = value;
    return u.f;
}

struct Parameter {
    const void* data;
    const bf16* get_bf16() const { return reinterpret_cast<const bf16*>(data); }
};

struct Activation {
    size_t size;
    std::unique_ptr<float[]> data;
    explicit Activation(size_t n) : size(n), data(new float[n]) {}
};

std::ostream& operator<<(std::ostream& out, const Activation& a) {
    if (a.size < 16) {
        for (auto i = 0u; i < a.size; ++i) {
            if (i) {
                out << ", ";
            }
            out << a.data[i];
        }
    } else {
        for (auto i : std::vector<size_t>({0, 1, 2, a.size - 3, a.size - 2, a.size - 1})) {
            if (i == a.size - 3) {
                out << " ... ";
            } else if (i) {
                out << ", ";
            }
            out << a.data[i];
        }
    }
    return out;
}

///////////////////////////////////////////////////////////////////////////////
// Model

struct Layer {
    Parameter attnNorm;
    Parameter attnQ;
    Parameter attnK;
    Parameter attnV;
    Parameter attnO;

    Parameter mlpNorm;
    Parameter mlpUp;
    Parameter mlpGate;
    Parameter mlpDown;
};

struct Model {
    unsigned nLayers;
    unsigned dModel;
    unsigned dFFN;
    unsigned dAttnHead;
    unsigned dAttnKV;
    unsigned dAttnQ;
    std::vector<float> ropeFreq;
    float normEps;

    Parameter embedTokens;
    std::vector<Layer> layers;
    Parameter finalNorm;

    std::vector<char> _parameterData;

    Model() = default;
    Model(const Model&) = delete;
    Model(Model&&) = default;
    Model& operator=(const Model&) = delete;
    Model& operator=(Model&&) = default;
};

Model loadConfig(std::istream& file) {
    auto config = json::parse(file);

    Model m;
    m.nLayers = config["num_hidden_layers"].template get<unsigned>();
    m.dModel = config["hidden_size"].template get<unsigned>();
    m.dFFN = config["intermediate_size"].template get<unsigned>();
    m.dAttnHead = config["head_dim"].template get<unsigned>();
    m.dAttnKV = config["num_key_value_heads"].template get<unsigned>();
    m.dAttnQ = config["num_attention_heads"].template get<unsigned>() / m.dAttnKV;
    m.normEps = config["rms_norm_eps"].template get<float>();

    auto theta = config["rope_theta"].template get<float>();
    auto scaling = config["rope_scaling"];
    auto factor = scaling["factor"].template get<float>();
    auto lowFreqFactor = scaling["low_freq_factor"].template get<float>();
    auto highFreqFactor = scaling["high_freq_factor"].template get<float>();
    auto originalLength = scaling["original_max_position_embeddings"].template get<unsigned>();

    for (auto i = 0u; i < m.dAttnHead; i += 2) {
        auto freq = std::pow(theta, -static_cast<float>(i) / m.dAttnHead);
        auto z = (originalLength * freq / (2 * static_cast<float>(M_PI)) - lowFreqFactor) /
                 (highFreqFactor - lowFreqFactor);
        z = std::clamp(z, 0.f, 1.f);
        m.ropeFreq.push_back(freq * ((1 - z) / factor + z));
    }
    return m;
}

void loadParameters(Model& model, std::istream& file) {
    uint64_t nHeader(0);
    file.read(reinterpret_cast<char*>(&nHeader), sizeof(nHeader));

    // Read the JSON header
    std::string headerData(nHeader, '\0');
    file.read(headerData.data(), headerData.size());
    auto header = json::parse(headerData);
    header.erase("__metadata__");
    uint64_t maxOffset(0);
    for (auto e : header.items()) {
        maxOffset = std::max(maxOffset, e.value()["data_offsets"][1].template get<uint64_t>());
    }

    // Read the data buffer, in chunks
    std::vector<char> tensorData;
    model._parameterData.reserve(maxOffset);
    constexpr uint64_t chunkSize(1 << 16);
    for (auto i = uint64_t(0); i < maxOffset; i += chunkSize) {
        file.read(model._parameterData.data() + i, std::min(chunkSize, maxOffset - i));
    }

    // Load the parameter pointers
    auto load = [&model, &header](const std::string& name) -> Parameter {
        auto j = header["model." + name + ".weight"];
        if (j["dtype"].template get<std::string>() != "BF16") {
            throw std::invalid_argument("Non-BF16 data");
        }
        auto start = j["data_offsets"][0].template get<uint64_t>();
        return {model._parameterData.data() + start};
    };
    model.embedTokens = load("embed_tokens");
    for (auto idx = 0u; idx < model.nLayers; ++idx) {
        auto pre = std::format("layers.{}.", idx);
        Layer layer;
        layer.attnNorm = load(pre + "input_layernorm");
        layer.attnQ = load(pre + "self_attn.q_proj");
        layer.attnK = load(pre + "self_attn.k_proj");
        layer.attnV = load(pre + "self_attn.v_proj");
        layer.attnO = load(pre + "self_attn.o_proj");
        layer.mlpNorm = load(pre + "post_attention_layernorm");
        layer.mlpGate = load(pre + "mlp.gate_proj");
        layer.mlpUp = load(pre + "mlp.up_proj");
        layer.mlpDown = load(pre + "mlp.down_proj");
        model.layers.push_back(layer);
    }
    model.finalNorm = load("norm");
}

///////////////////////////////////////////////////////////////////////////////
// Ops

Activation embeddingLookup(const std::vector<unsigned>& tokens,
                           const bf16* weight,
                           unsigned dModel) {
    Activation y(tokens.size() * dModel);
    for (auto n = 0u; n < tokens.size(); ++n) {
        for (auto i = 0u; i < dModel; ++i) {
            y.data[n * dModel + i] = bf16_to_float(weight[tokens[n] * dModel + i]);
        }
    }
    return y;
}

Activation rmsNorm(const Activation& x, const bf16* weight, unsigned dModel, float eps) {
    Activation y(x.size);
    for (auto i0 = 0u; i0 < x.size; i0 += dModel) {
        float sumSq = 0;
        for (auto i = 0u; i < dModel; ++i) {
            sumSq += x.data[i0 + i] * x.data[i0 + i];
        }
        float norm = 1 / (std::sqrt(sumSq / dModel + eps));
        for (auto i = 0u; i < dModel; ++i) {
            y.data[i0 + i] = x.data[i0 + i] * norm * bf16_to_float(weight[i]);
        }
    }
    return y;
}

Activation project(const Activation& x, const bf16* weight, unsigned dIn, unsigned dOut) {
    Activation y(x.size / dIn * dOut);
    for (auto n = 0u; n < x.size / dIn; ++n) {
        for (auto j = 0u; j < dOut; ++j) {
            float dot = 0;
            for (auto i = 0u; i < dIn; ++i) {
                dot += x.data[n * dIn + i] * bf16_to_float(weight[j * dIn + i]);
            }
            y.data[n * dOut + j] = dot;
        }
    }
    return y;
}

///////////////////////////////////////////////////////////////////////////////
// Functions

void predict(const Model& model, const std::vector<unsigned>& tokens) {
    auto embedding = embeddingLookup(tokens, model.embedTokens.get_bf16(), model.dModel);

    // [0].attn
    auto z = rmsNorm(embedding, model.layers[0].attnNorm.get_bf16(), model.dModel, model.normEps);
    auto q = project(z, model.layers[0].attnQ.get_bf16(), model.dModel,
                     model.dAttnKV * model.dAttnQ * model.dAttnHead);
    auto k =
        project(z, model.layers[0].attnK.get_bf16(), model.dModel, model.dAttnKV * model.dAttnHead);
    auto v =
        project(z, model.layers[0].attnV.get_bf16(), model.dModel, model.dAttnKV * model.dAttnHead);
    std::cerr << "q: " << q << std::endl;
    std::cerr << "k: " << k << std::endl;
    std::cerr << "v: " << v << std::endl;
}

}  // namespace lp

int main(int argc, char** argv) {
    if (argc < 3) {
        throw std::runtime_error(
            "Not enough arguments."
            " Usage: ./model path/to/config.json path/to/model.safetensors");
    }

    std::ifstream configFile(argv[1]);
    auto model = lp::loadConfig(configFile);
    std::ifstream dataFile(argv[2]);
    lp::loadParameters(model, dataFile);

    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream lineS(line);
        std::vector<unsigned> tokens;
        while (lineS.good()) {
            unsigned token;
            if (lineS >> token) {
                tokens.push_back(token);
            }
        }
        lp::predict(model, tokens);
    }

    return 0;
}
