/*
  reprocess_dataset <dataset_folder>
                    --target-stage <0-3>
                    [--overwrite | --no-overwrite]
                    [--rc-version <n>]
                    [--ld-version <n>]
                    [--out <out_path>]

  Reprocesses all frames in a dataset through the full pipeline
  (prefiltering -> RC -> LD) up to target_stage.

  target_stage: 0=NotPrefiltered  1=Prefiltered  2=RCNeted  3=LDNeted

  Frames whose stored InferenceResults already match the requested model
  versions and config are skipped. If --overwrite is set, frames with
  different conditions are reprocessed; otherwise they are left as-is.

  After processing, target_processing_stage and frame statistics in
  dataset.json are updated to reflect the current state of all frames.

  Optional rc_version and ld_version default to 2 if not supplied.
  LDNetConfig defaults match the standard FP16 TRT configuration.

  --out  File to write the processing metadata path into. Falls back to
         path.out if not provided or not writable.
*/

#include <fstream>
#include <string>

#include <CLI/CLI.hpp>
#include "spdlog/spdlog.h"
#include "inference/inference_manager.hpp"
#include "inference/types.hpp"
#include "vision/dataset.hpp"
#include "vision/frame.hpp"
#include "vision/reprocessing.hpp"
#include "core/data_handling.hpp"

static constexpr const char* kDefaultOutPath      = "path.out";

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

int main(int argc, char** argv)
{
    spdlog::set_level(spdlog::level::info);

    CLI::App app{"Reprocess all frames in a dataset."};
    app.allow_extras(false);

    std::string dataset_folder;
    int target_stage = 3;
    bool overwrite = false;
    int rc_version = 2;
    int ld_version = 2;
    std::string out_path = kDefaultOutPath;

    app.add_option("dataset_folder", dataset_folder, "Path to the dataset folder")->required();
    app.add_option("--target-stage", target_stage,
                   "Target stage: 0=NotPrefiltered 1=Prefiltered 2=RCNeted 3=LDNeted")
        ->required()
        ->check(CLI::Range(0, 3));
    app.add_flag("--overwrite,--no-overwrite{false}", overwrite,
                 "Reprocess frames whose stored inference results do not match the requested conditions");
    app.add_option("--rc-version", rc_version, "RCNet model version");
    app.add_option("--ld-version", ld_version, "LDNet model version");
    app.add_option("--out", out_path, "File to write the processing metadata path into");

    CLI11_PARSE(app, argc, argv);

    const ProcessingStage target = ToProcessingStage(static_cast<uint8_t>(target_stage));
    const std::string resolved_out_path = ResolveOutPath(out_path);

    spdlog::info("reprocess_dataset: folder={} target={} overwrite={} rc_version={} ld_version={}",
                 dataset_folder, ProcessingStageToString(target), overwrite, rc_version, ld_version);

    Dataset dataset(dataset_folder);

    InferenceManager im;

    EC ec = im.SetRCNetVersion(rc_version);
    if (ec != EC::OK)
    {
        spdlog::error("Failed to set RC version {}: error {}", rc_version, to_uint8(ec));
        return to_uint8(ec);
    }

    ec = im.SetLDNetVersion(ld_version);
    if (ec != EC::OK)
    {
        spdlog::error("Failed to set LD version {}: error {}", ld_version, to_uint8(ec));
        return to_uint8(ec);
    }

    im.SetLDNetConfig(NET_QUANTIZATION::FP16, 4608, 2592, false, true);

    ec = Reprocessing::Dataset(dataset, im, target, overwrite);
    if (ec != EC::OK)
    {
        spdlog::error("Reprocessing::Dataset returned error {}", to_uint8(ec));
        return to_uint8(ec);
    }

    std::string processing_file_path = DH::StoreProcessingMetadataToDisk(dataset);

    SPDLOG_INFO("Dataset reprocessing completed. Dataset stored in folder: {}. Processing file path {} stored in {}",
                dataset_folder, processing_file_path, resolved_out_path);
    WriteResult(resolved_out_path, processing_file_path);
    SPDLOG_INFO("reprocess_dataset: complete");
    return 0;
}
