#ifndef INFERENCE_RESULTS_HPP
#define INFERENCE_RESULTS_HPP

#include <vector>
#include "inference/types.hpp"
#include "vision/regions.hpp"
#include "vision/ld.hpp"
#include <nlohmann/json.hpp>

struct InferenceResults {
    int rc_version = -1;
    int ld_version = -1;
    LDNetConfig ldnet_config{};
    std::vector<Region>   regions;
    std::vector<Landmark> landmarks;

    nlohmann::ordered_json toJson() const;
    static InferenceResults fromJson(const nlohmann::json& j);
};

#endif // INFERENCE_RESULTS_HPP
