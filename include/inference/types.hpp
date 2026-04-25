#ifndef INFERENCE_TYPES_HPP
#define INFERENCE_TYPES_HPP

#include <string>
#include <cstdint>

namespace Inference
{

enum class NET_QUANTIZATION : uint8_t
{
    FP32 = 0,
    FP16 = 1,
    INT8 = 2,
};

inline std::string GetQuantString(NET_QUANTIZATION quant) {
    switch (quant) {
        case NET_QUANTIZATION::FP32: return "fp32";
        case NET_QUANTIZATION::FP16: return "fp16";
        case NET_QUANTIZATION::INT8: return "int8";
        default: return "";
    }
}

// Path helpers: derive model paths from a version integer.
// RC:  models/trained-rc/V{N}/rc_model_weights.trt
// LD:  models/trained-ld/V{N}/
inline std::string RCEnginePath(int version) {
    return "./models/trained-rc/V" + std::to_string(version) + "/rc_model_weights.trt";
}

inline std::string LDFolderPath(int version) {
    return "./models/trained-ld/V" + std::to_string(version);
}

// Parameters used to define the LD model file name (from region + config)
struct LDNetConfig {
    NET_QUANTIZATION weight_quant;
    int input_width;
    int input_height;
    bool embedded_nms;
    bool use_trt;

    bool operator==(const LDNetConfig& other) const {
        return weight_quant  == other.weight_quant  &&
               input_width   == other.input_width   &&
               input_height  == other.input_height  &&
               embedded_nms  == other.embedded_nms  &&
               use_trt       == other.use_trt;
    }
    bool operator!=(const LDNetConfig& other) const { return !(*this == other); }

    std::string GetFileNameAppendix() {
        std::string fp16_string = GetQuantString(weight_quant);
        std::string nms_string = embedded_nms ? "_nms" : "";
        std::string file_ext = use_trt ? "trt" : "onnx";
        std::string size_str = (input_height == input_width)
            ? std::to_string(input_width)
            : std::to_string(input_height) + "x" + std::to_string(input_width);
        return "_weights_" + fp16_string + "_sz_" + size_str + nms_string + "." + file_ext;
    }
};

} // namespace Inference

// Aliases for code that used the old unqualified names
using NET_QUANTIZATION = Inference::NET_QUANTIZATION;
using LDNetConfig = Inference::LDNetConfig;

#endif // INFERENCE_TYPES_HPP
