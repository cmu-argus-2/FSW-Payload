#include "inference/inference_results.hpp"
#include "spdlog/spdlog.h"

namespace {

Inference::NET_QUANTIZATION QuantFromString(const std::string& s)
{
    if (s == "fp16") return Inference::NET_QUANTIZATION::FP16;
    if (s == "int8") return Inference::NET_QUANTIZATION::INT8;
    return Inference::NET_QUANTIZATION::FP32;
}

nlohmann::ordered_json LDNetConfigToJson(const LDNetConfig& cfg)
{
    nlohmann::ordered_json j;
    j["weight_quant"] = Inference::GetQuantString(cfg.weight_quant);
    j["input_width"]  = cfg.input_width;
    j["input_height"] = cfg.input_height;
    j["embedded_nms"] = cfg.embedded_nms;
    j["use_trt"]      = cfg.use_trt;
    return j;
}

LDNetConfig LDNetConfigFromJson(const nlohmann::json& j)
{
    LDNetConfig cfg{};
    cfg.weight_quant = QuantFromString(j.value("weight_quant", std::string("fp16")));
    cfg.input_width  = j.value("input_width",  0);
    cfg.input_height = j.value("input_height", 0);
    cfg.embedded_nms = j.value("embedded_nms", false);
    cfg.use_trt      = j.value("use_trt",      true);
    return cfg;
}

} // namespace

nlohmann::ordered_json InferenceResults::toJson() const
{
    nlohmann::ordered_json j;
    j["rcnet_version"]            = rc_version;
    j["ldnet_version"]            = ld_version;
    j["ldnet_config"]             = LDNetConfigToJson(ldnet_config);
    j["detected_regions_count"]   = regions.size();
    j["detected_landmarks_count"] = landmarks.size();

    j["regions"] = nlohmann::ordered_json::array();
    for (size_t i = 0; i < regions.size(); ++i)
    {
        nlohmann::ordered_json item;
        item["region_" + std::to_string(i)] = {
            {"id",         regions[i].id},
            {"confidence", regions[i].confidence}
        };
        j["regions"].push_back(item);
    }

    j["landmarks"] = nlohmann::ordered_json::array();
    for (size_t i = 0; i < landmarks.size(); ++i)
    {
        const auto& lm = landmarks[i];
        nlohmann::ordered_json item;
        item["landmark_" + std::to_string(i)] = {
            {"x",          lm.x},
            {"y",          lm.y},
            {"height",     lm.height},
            {"width",      lm.width},
            {"confidence", lm.confidence},
            {"class_id",   lm.class_id},
            {"region_id",  lm.region_id}
        };
        j["landmarks"].push_back(item);
    }
    return j;
}

InferenceResults InferenceResults::fromJson(const nlohmann::json& j)
{
    InferenceResults r;
    try {
        r.rc_version = j.value("rcnet_version", -1);
        r.ld_version = j.value("ldnet_version", -1);

        if (j.contains("ldnet_config") && j.at("ldnet_config").is_object())
            r.ldnet_config = LDNetConfigFromJson(j.at("ldnet_config"));

        if (j.contains("regions") && j.at("regions").is_array())
        {
            for (const auto& item : j.at("regions"))
            {
                for (const auto& [key, val] : item.items())
                {
                    RegionID rid  = static_cast<RegionID>(val.at("id").get<int>());
                    float    conf = val.at("confidence").get<float>();
                    r.regions.emplace_back(rid, conf);
                }
            }
        }

        if (j.contains("landmarks") && j.at("landmarks").is_array())
        {
            for (const auto& item : j.at("landmarks"))
            {
                for (const auto& [key, val] : item.items())
                {
                    float    x         = val.at("x").get<float>();
                    float    y         = val.at("y").get<float>();
                    float    h         = val.at("height").get<float>();
                    float    w         = val.at("width").get<float>();
                    float    conf      = val.at("confidence").get<float>();
                    uint16_t class_id  = val.at("class_id").get<uint16_t>();
                    RegionID region_id = static_cast<RegionID>(val.at("region_id").get<int>());
                    r.landmarks.emplace_back(x, y, class_id, region_id, h, w, conf);
                }
            }
        }
    } catch (const std::exception& e) {
        SPDLOG_ERROR("InferenceResults::fromJson failed: {}", e.what());
    }
    return r;
}
