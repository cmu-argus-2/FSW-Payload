/*
  reprocess_dataset <dataset_folder> <target_stage:0-3> <overwrite:0|1> [rc_version] [ld_version]

  Reprocesses all frames in a dataset through the full pipeline
  (prefiltering -> RC -> LD) up to target_stage.

  target_stage: 0=NotPrefiltered  1=Prefiltered  2=RCNeted  3=LDNeted

  Frames whose stored InferenceResults already match the requested model
  versions and config are skipped. If overwrite=1, frames with different
  conditions are reprocessed; if overwrite=0 they are left as-is.

  After processing, target_processing_stage and frame statistics in
  dataset.json are updated to reflect the current state of all frames.

  Optional rc_version and ld_version default to 2 if not supplied.
  LDNetConfig defaults match the standard FP16 TRT configuration.
*/

#include <string>
#include "spdlog/spdlog.h"
#include "inference/inference_manager.hpp"
#include "inference/types.hpp"
#include "vision/dataset.hpp"
#include "vision/frame.hpp"
#include "vision/reprocessing.hpp"
#include "core/data_handling.hpp"


static constexpr const char* kDefaultOutPath      = "path.out";

static void WriteResult(const std::string& out_file, const std::string& content)
{
    std::ofstream f(out_file, std::ios::trunc);
    if (!f.is_open()) { spdlog::error("Failed to write result to '{}'", out_file); return; }
    f << content << '\n';
}

int main(int argc, char** argv)
{
    if (argc < 4)
    {
        spdlog::error("Usage: reprocess_dataset <dataset_folder> <target_stage:0-3> <overwrite:0|1> [rc_version] [ld_version]");
        return 1;
    }

    const std::string dataset_folder = argv[1];
    const ProcessingStage target     = ToProcessingStage(static_cast<uint8_t>(std::stoi(argv[2])));
    const bool overwrite             = std::string(argv[3]) != "0";
    const int rc_version             = (argc > 4) ? std::stoi(argv[4]) : 2;
    const int ld_version             = (argc > 5) ? std::stoi(argv[5]) : 2;

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

    SPDLOG_INFO("Dataset collection completed. Dataset stored in folder: {}. Processing File Path {}stored in {}", dataset_folder, processing_file_path, kDefaultOutPath);
    WriteResult(kDefaultOutPath, dataset_folder + "/processing.json");
    SPDLOG_INFO("reprocess_dataset: complete");
    return 0;
}
