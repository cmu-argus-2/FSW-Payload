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

// Parameters used to define the LD model file name (from region + config)
struct LDNetConfig {
    NET_QUANTIZATION weight_quant;
    int input_width;
    int input_height;
    bool embedded_nms;
    bool use_trt;

    std::string GetFileNameAppendix() {
        std::string fp16_string = GetQuantString(weight_quant);
        std::string nms_string = embedded_nms ? "_nms" : "";
        std::string file_ext = use_trt ? "trt" : "onnx";
        return "_weights_" + fp16_string + "_sz_" + std::to_string(input_width) + nms_string + "." + file_ext;
    }
};

} // namespace Inference

// Aliases for code that used the old unqualified names
using NET_QUANTIZATION = Inference::NET_QUANTIZATION;
using LDNetConfig = Inference::LDNetConfig;

#endif // INFERENCE_TYPES_HPP
