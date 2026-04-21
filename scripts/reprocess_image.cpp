/*
  reprocess_image <raw_image_path> <target_stage:0-3> <overwrite:0|1> [rc_version] [ld_version]

  Reprocesses a single raw image from data/images/ through the full pipeline
  (prefiltering -> RC -> LD) up to the specified target stage.

  target_stage: 0=NotPrefiltered  1=Prefiltered  2=RCNeted  3=LDNeted

  Frames whose stored InferenceResults already match the requested model
  versions and config are skipped. If overwrite=1, frames with different
  conditions are reprocessed; if overwrite=0 they are left as-is.

  Optional rc_version and ld_version default to 2 if not supplied.
  LDNetConfig defaults match the standard FP16 TRT configuration.
*/

#include <string>
#include "spdlog/spdlog.h"
#include "inference/inference_manager.hpp"
#include "inference/types.hpp"
#include "vision/frame.hpp"
#include "vision/reprocessing.hpp"

int main(int argc, char** argv)
{
    if (argc < 4)
    {
        spdlog::error("Usage: reprocess_image <raw_image_path> <target_stage:0-3> <overwrite:0|1> [rc_version] [ld_version]");
        return 1;
    }

    const std::string image_path   = argv[1];
    const ProcessingStage target   = ToProcessingStage(static_cast<uint8_t>(std::stoi(argv[2])));
    const bool overwrite           = std::string(argv[3]) != "0";
    const int rc_version           = (argc > 4) ? std::stoi(argv[4]) : 2;
    const int ld_version           = (argc > 5) ? std::stoi(argv[5]) : 2;

    spdlog::info("reprocess_image: path={} target={} overwrite={} rc_version={} ld_version={}",
                 image_path, ProcessingStageToString(target), overwrite, rc_version, ld_version);

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

    ec = Reprocessing::Image(image_path, im, target, overwrite);
    if (ec != EC::OK)
    {
        spdlog::error("Reprocessing::Image returned error {}", to_uint8(ec));
        return to_uint8(ec);
    }

    spdlog::info("reprocess_image: complete");
    return 0;
}
