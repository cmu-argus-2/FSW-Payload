#include <memory>
#include <filesystem>
#include "spdlog/spdlog.h"
#include "vision/frame.hpp"
#include "core/data_handling.hpp"
#include "core/timing.hpp"
#include "inference/orchestrator.hpp"
#include "vision/regions.hpp"
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <fstream>

using namespace cv;
using namespace cv::dnn;


/*
    This file is a simple test for the inference orchestrator.
    It grabs a sample image from disk, initializes the orchestrator,
    and runs inference on the image to fill the regions and landmarks.
*/

void drawPrediction(int classId, float conf, int left, int top, int right, int bottom, Mat& frame, Scalar color)
{
    rectangle(frame, Point(left, top), Point(right, bottom), color);

    std::string label = format("%.2f", conf);

    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - labelSize.height),
              Point(left + labelSize.width, top + baseLine), Scalar::all(255), FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar());
}


static inline cv::Rect scaleBoxBackLetterbox(
    const cv::Rect& rBlob,
    const cv::Size& imgSize,
    const cv::Size& netSize)
{
    const float gain = std::min(netSize.width / (float)imgSize.width,
                                netSize.height / (float)imgSize.height);
    const float padX = (netSize.width  - imgSize.width  * gain) * 0.5f;
    const float padY = (netSize.height - imgSize.height * gain) * 0.5f;

    float x = (rBlob.x - padX) / gain;
    float y = (rBlob.y - padY) / gain;
    float w = rBlob.width  / gain;
    float h = rBlob.height / gain;

    cv::Rect r((int)std::round(x), (int)std::round(y),
               (int)std::round(w), (int)std::round(h));
    return r & cv::Rect(0, 0, imgSize.width, imgSize.height);
}

int main(int argc, char** argv)
{
    std::string rc_trt_file_path;
    std::string ld_trt_folder_path;
    std::string sample_image_path;
    std::string_view target_folder;
    if (argc < 5)
    {
        spdlog::info("Using default inference example");
        rc_trt_file_path = "models/V1/trained-rc/effnet_0997acc.trt";
        ld_trt_folder_path = "models/V1/trained-ld";
        target_folder = "data/images";
        // ld_trt_folder_path -> should be inferred from the parameters
        std::string tgt_region = "17T";
        std::string sample_id = "00277";
        bool isjpg = false;
        sample_image_path = "models/V1/sample_images/l8_" + tgt_region + "_" + sample_id;
        if (isjpg) {
            sample_image_path = sample_image_path + ".jpg";
        } else {
            sample_image_path = sample_image_path + ".png";
        }
    } else {
        rc_trt_file_path  = argv[1];
        ld_trt_folder_path = argv[2];
        sample_image_path  = argv[3];
        target_folder      = argv[4];
    }
    // Parameters that define the model
    // TRT Example
    Inference::NET_QUANTIZATION weight_quant = Inference::NET_QUANTIZATION::FP16;
    int input_width = 4608;
    int input_height = 2592;
    bool embedded_nms = false;
    bool use_trt_for_ld = true;
    // ONNX example
    // Inference::NET_QUANTIZATION weight_quant = Inference::NET_QUANTIZATION::FP32;
    // int input_width = 4608;
    // int input_height = 4608;
    // bool embedded_nms = false;
    // bool use_trt_for_ld = false;

    Inference::Orchestrator orchestrator;
    orchestrator.SetRCNetEnginePath(rc_trt_file_path);
    orchestrator.SetLDNetConfig(weight_quant, input_width, input_height, embedded_nms, use_trt_for_ld);
    orchestrator.SetLDNetEngineFolderPath(ld_trt_folder_path);
    // orchestrator.Initialize(rc_trt_file_path, ld_trt_folder_path);

    spdlog::info("Using image file: {}", sample_image_path);

    Frame frame; // empty frame 
    auto timestamp = timing::GetCurrentTimeMs();
    if (!DH::ReadImageFromDisk(sample_image_path, frame, 0,  static_cast<uint64_t>(timestamp)))
    {
        spdlog::error("Failed to read image from disk: {}", sample_image_path);
        return 1;
    }

    std::shared_ptr<Frame> frame_ptr = std::make_shared<Frame>(frame);

    orchestrator.GrabNewImage(frame_ptr); 

    spdlog::info("Running inference on the frame...");
    EC status = orchestrator.ExecFullInference();
    if (status != EC::OK)
    {
        spdlog::error("Inference failed with error code: {}", to_uint8(status));
        return 1;
    }
    spdlog::info("Inference completed successfully.");
    
    spdlog::info("Regions found: {}", frame_ptr->GetRegionIDs().size());

    for (const auto& region_id : frame_ptr->GetRegionIDs())
    {
        spdlog::info("Region ID: {}", GetRegionString(region_id));
    }

    DH::StoreFrameMetadataToDisk(*frame_ptr, target_folder); // Test for now
    spdlog::info("Frame metadata JSON saved to data/images/");

    // Landmark Detection results
    std::vector<Landmark> landmarks = frame_ptr->GetLandmarks();
    spdlog::info("Landmarks found: {}", landmarks.size());
    for (const auto& landmark : landmarks)
    {
        spdlog::info("Landmark - Class ID: {}, Region ID: {}, Confidence: {:.3f}, Position: ({:.2f}, {:.2f}), Size: ({:.2f}, {:.2f})",
            landmark.class_id, GetRegionString(landmark.region_id), landmark.confidence, landmark.x, landmark.y, landmark.height, landmark.width);
    }
    
    // TODO: Look for label file if there is one. If there is, cross-validate
    std::string label_file_path = sample_image_path.substr(0, sample_image_path.find_last_of('.')) + ".txt";
    if (std::filesystem::exists(label_file_path))
    {
        spdlog::info("Label file found: {}", label_file_path);
        std::vector<Rect> boxes;
        cv::Mat img = cv::imread(sample_image_path, cv::IMREAD_COLOR);
        Size size = Size(4608, 4608);
        for (const auto& landmark : landmarks)
        {
            boxes.push_back(Rect(cvFloor(landmark.x - landmark.width/2), cvFloor(landmark.y - landmark.height/2), cvFloor(landmark.width), cvFloor(landmark.height)));
        }
        spdlog::info("Converted {} landmarks to Rect", boxes.size());

        // for (auto& b : boxes) {
        //     b = scaleBoxBackLetterbox(b, img.size(), size);
        // }
        // spdlog::info("Boxes scaled back to original image");

        for (size_t idx = 0; idx < boxes.size(); ++idx)
        {
            Rect box = boxes[idx];
            Landmark landmark = landmarks[idx];
            drawPrediction(landmark.class_id, landmark.confidence, box.x, box.y,
                    box.width + box.x, box.height + box.y, img, Scalar(0, 255, 0));
        }
        spdlog::info("Drew {} predictions", boxes.size());

        // Load the labels
        std::string labels_path = sample_image_path;
        size_t dot_pos = labels_path.find_last_of(".");
        if (dot_pos != std::string::npos) {
            labels_path = labels_path.substr(0, dot_pos) + ".txt";
        }
        std::ifstream labels_file(labels_path);
        std::vector<Rect> gt_boxes;
        std::string line;
        while (std::getline(labels_file, line)) {
            std::istringstream iss(line);
            int class_id;
            float nx, ny, nw, nh;
            if (iss >> class_id >> nx >> ny >> nw >> nh) {
                int x = static_cast<int>(nx * img.cols - nw * img.cols / 2);
                int y = static_cast<int>(ny * img.rows - nh * img.rows / 2);
                int w = static_cast<int>(nw * img.cols);
                int h = static_cast<int>(nh * img.rows);
                gt_boxes.push_back(Rect(x, y, w, h));
                drawPrediction(class_id, 1.0f, x, y, x + w, y + h, img, Scalar(255, 0, 0));
            }
            // look for classid in boxes 
            // if found, compute iou between gt box and pred box, if iou > 0.5, count as true positive, else count as false positive
            // if not found, count as false negative
            for (size_t idx = 0; idx < boxes.size(); ++idx)
            {
                if (landmarks[idx].class_id != class_id) {
                    continue;
                }
                Rect pred_box = boxes[idx];
                float iou = (pred_box & gt_boxes.back()).area() / float((pred_box | gt_boxes.back()).area());
                if (iou > 0.5) {
                    spdlog::info("True positive: class_id={}, iou={:.2f}", landmarks[idx].class_id, iou);
                } else {
                    spdlog::info("False positive: class_id={}, iou={:.2f}", landmarks[idx].class_id, iou);
                }
            }
        }
        spdlog::info("Loaded {} ground truth boxes", gt_boxes.size());

        cv::imwrite("debug.jpg", img);
        spdlog::info("Image displayed");

    }
    else
    {
        spdlog::warn("No label file found.");
    }

    return 0;
}