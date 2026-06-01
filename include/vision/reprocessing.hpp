#ifndef REPROCESSING_HPP
#define REPROCESSING_HPP

#include <string>
#include "vision/dataset.hpp"
#include "inference/inference_manager.hpp"
#include "vision/frame.hpp"
#include "core/errors.hpp"

namespace Reprocessing
{

// Reprocess all frames in a dataset using the current InferenceManager configuration.
// target is the processing stage to reach; the dataset's target_processing_stage is
// updated to match and dataset.json is written to disk when the function returns.
// Same conditions (versions + ldnet_config) always reuse stored results.
// Different conditions reprocess only if overwrite = true.
EC Dataset(::Dataset& dataset, InferenceManager& im, ProcessingStage target, bool overwrite,
           bool bypass_prefilter_rejection = false);

// Reprocess a single image file. Dataset-style raw_<timestamp>_<camid>.{jpg|png}
// names reuse matching frame metadata when present. Images without matching
// metadata are processed as fresh frames; non-dataset filenames use current time
// and camera 0. Same conditions always reuse existing metadata; different
// conditions reprocess only if overwrite = true.
EC Image(const std::string& raw_image_path, InferenceManager& im,
         ProcessingStage target, bool overwrite,
         bool bypass_prefilter_rejection = false,
         std::string* metadata_path_out = nullptr);

} // namespace Reprocessing

#endif // REPROCESSING_HPP
