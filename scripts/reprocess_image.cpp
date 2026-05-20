/*
  reprocess_image <raw_image_path>
                  --target-stage <0-3>
                  [--overwrite | --no-overwrite]
                  [--rc-version <n>]
                  [--ld-version <n>]
                  [--ld-weight-quant <0|1|2>]
                  [--ld-input-width <px>]
                  [--ld-input-height <px>]
                  [--ld-embedded-nms | --no-ld-embedded-nms]
                  [--ld-use-trt | --no-ld-use-trt]
                  [--system-config <path>]
                  [--bypass-prefilter-rejection]
                  [--out <out_path>]

  Reprocesses a single raw image from data/images/ through the full pipeline
  (prefiltering -> RC -> LD) up to the specified target stage.

  target_stage: 0=NotPrefiltered  1=Prefiltered  2=RCNeted  3=LDNeted

  Frames whose stored InferenceResults already match the requested model
  versions and config are skipped. If --overwrite is set, frames with
  different conditions are reprocessed; otherwise they are left as-is.

  RC/LD defaults come from config/config.toml when present, falling back to
  InferenceManager defaults. CLI version/config flags override both.

  --out  File to write the generated frame metadata path into. Falls back to
         path.out if not provided or not writable.
*/

#include <fstream>
#include <optional>
#include <string>

#include <CLI/CLI.hpp>
#include "spdlog/spdlog.h"
#include "configuration.hpp"
#include "inference/inference_manager.hpp"
#include "inference/types.hpp"
#include "vision/frame.hpp"
#include "vision/reprocessing.hpp"

static constexpr const char* kDefaultSystemConfigPath = "config/config.toml";
static constexpr const char* kDefaultOutPath = "path.out";

static std::string ResolveOutPath(const std::string& path)
{
    if (path == kDefaultOutPath) return path;
    std::ofstream f(path, std::ios::app);
    if (f.is_open()) return path;
    spdlog::warn("Cannot write to '{}', using default output '{}'", path, kDefaultOutPath);
    return kDefaultOutPath;
}

static void WriteResult(const std::string& out_file, const std::string& content)
{
    std::ofstream f(out_file, std::ios::trunc);
    if (!f.is_open()) { spdlog::error("Failed to write result to '{}'", out_file); return; }
    f << content << '\n';
}

static EC ApplyInferenceCliOverrides(InferenceManager& inference_manager,
                                     const std::optional<int>& rc_version,
                                     const std::optional<int>& ld_version,
                                     const std::optional<int>& ld_weight_quant,
                                     const std::optional<int>& ld_input_width,
                                     const std::optional<int>& ld_input_height,
                                     const std::optional<bool>& ld_embedded_nms,
                                     const std::optional<bool>& ld_use_trt)
{
    if (rc_version) {
        EC ec = inference_manager.SetRCNetVersion(*rc_version);
        if (ec != EC::OK) return ec;
    }

    if (ld_version) {
        EC ec = inference_manager.SetLDNetVersion(*ld_version);
        if (ec != EC::OK) return ec;
    }

    LDNetConfig ldnet_config = inference_manager.GetLDNetConfig();
    bool ldnet_changed = false;

    if (ld_weight_quant) {
        ldnet_config.weight_quant = static_cast<NET_QUANTIZATION>(*ld_weight_quant);
        ldnet_changed = true;
    }
    if (ld_input_width) {
        ldnet_config.input_width = *ld_input_width;
        ldnet_changed = true;
    }
    if (ld_input_height) {
        ldnet_config.input_height = *ld_input_height;
        ldnet_changed = true;
    }
    if (ld_embedded_nms) {
        ldnet_config.embedded_nms = *ld_embedded_nms;
        ldnet_changed = true;
    }
    if (ld_use_trt) {
        ldnet_config.use_trt = *ld_use_trt;
        ldnet_changed = true;
    }

    if (ldnet_changed) {
        inference_manager.SetLDNetConfig(ldnet_config.weight_quant,
                                         ldnet_config.input_width,
                                         ldnet_config.input_height,
                                         ldnet_config.embedded_nms,
                                         ldnet_config.use_trt);
    }

    return EC::OK;
}

int main(int argc, char** argv)
{
    spdlog::set_level(spdlog::level::info);

    CLI::App app{"Reprocess a single raw image."};
    app.allow_extras(false);

    std::string image_path;
    int target_stage = 3;
    bool overwrite = false;
    std::optional<int> rc_version;
    std::optional<int> ld_version;
    std::optional<int> ld_weight_quant;
    std::optional<int> ld_input_width;
    std::optional<int> ld_input_height;
    bool ld_embedded_nms = false;
    bool ld_use_trt = false;
    bool bypass_prefilter_rejection = false;
    std::string system_config_path = kDefaultSystemConfigPath;
    std::string out_path = kDefaultOutPath;

    app.add_option("raw_image_path", image_path, "Path to the raw image")->required();
    app.add_option("--target-stage", target_stage,
                   "Target stage: 0=NotPrefiltered 1=Prefiltered 2=RCNeted 3=LDNeted")
        ->required()
        ->check(CLI::Range(0, 3));
    app.add_flag("--overwrite,--no-overwrite{false}", overwrite,
                 "Reprocess the frame when stored inference results do not match the requested conditions");
    app.add_option("--rc-version", rc_version, "RCNet model version");
    app.add_option("--ld-version", ld_version, "LDNet model version");
    app.add_option("--ld-weight-quant", ld_weight_quant, "LDNet weight quantization: 0=FP32 1=FP16 2=INT8")
        ->check(CLI::Range(0, 2));
    app.add_option("--ld-input-width", ld_input_width, "LDNet input width in pixels")
        ->check(CLI::PositiveNumber);
    app.add_option("--ld-input-height", ld_input_height, "LDNet input height in pixels")
        ->check(CLI::PositiveNumber);
    CLI::Option* ld_embedded_nms_opt =
        app.add_flag("--ld-embedded-nms,--no-ld-embedded-nms{false}", ld_embedded_nms,
                     "Use LDNet engines with embedded NMS");
    CLI::Option* ld_use_trt_opt =
        app.add_flag("--ld-use-trt,--no-ld-use-trt{false}", ld_use_trt,
                     "Use TensorRT LDNet engines instead of ONNX");
    app.add_option("--system-config", system_config_path, "System config TOML");
    app.add_flag("--bypass-prefilter-rejection", bypass_prefilter_rejection,
                 "Run prefiltering but continue to inference when prefiltering rejects the frame");
    app.add_option("--out", out_path, "File to write the frame metadata path into");

    CLI11_PARSE(app, argc, argv);

    const ProcessingStage target = ToProcessingStage(static_cast<uint8_t>(target_stage));
    const std::string resolved_out_path = ResolveOutPath(out_path);

    spdlog::info("reprocess_image: path={} target={} overwrite={}",
                 image_path, ProcessingStageToString(target), overwrite);

    InferenceManager im;
    Configuration config;
    try {
        config.LoadConfiguration(system_config_path);
    } catch (const std::exception& e) {
        spdlog::error("Failed to load system configuration {}: {}", system_config_path, e.what());
        return to_uint8(EC::PLACEHOLDER);
    }

    EC ec = config.ApplyInferenceConfig(im);
    if (ec != EC::OK) return to_uint8(ec);

    const std::optional<bool> ld_embedded_nms_override =
        ld_embedded_nms_opt->count() > 0 ? std::optional<bool>(ld_embedded_nms) : std::nullopt;
    const std::optional<bool> ld_use_trt_override =
        ld_use_trt_opt->count() > 0 ? std::optional<bool>(ld_use_trt) : std::nullopt;

    ec = ApplyInferenceCliOverrides(im,
                                    rc_version,
                                    ld_version,
                                    ld_weight_quant,
                                    ld_input_width,
                                    ld_input_height,
                                    ld_embedded_nms_override,
                                    ld_use_trt_override);
    if (ec != EC::OK) return to_uint8(ec);

    std::string metadata_path;
    ec = Reprocessing::Image(image_path, im, target, overwrite, bypass_prefilter_rejection, &metadata_path);
    if (ec != EC::OK)
    {
        spdlog::error("Reprocessing::Image returned error {}", to_uint8(ec));
        return to_uint8(ec);
    }

    WriteResult(resolved_out_path, metadata_path);
    spdlog::info("reprocess_image: frame metadata path {} stored in {}", metadata_path, resolved_out_path);
    spdlog::info("reprocess_image: complete");
    return 0;
}
