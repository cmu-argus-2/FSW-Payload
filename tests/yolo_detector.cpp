/**
 * @file yolo_detector.cpp
 * @brief Yolo Object Detection Sample
 * @author OpenCV team ( modified to work with Argus)
 */

//![includes]
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <fstream>
#include <sstream>
#include "iostream"
#include "spdlog/spdlog.h"
#include <opencv2/highgui.hpp>
//![includes]

using namespace cv;
using namespace cv::dnn;

void getClasses(std::string classesFile);
void drawPrediction(int classId, float conf, int left, int top, int right, int bottom, Mat& frame, Scalar color);
void yoloPostProcessing(
    std::vector<Mat>& outs,
    std::vector<int>& keep_classIds,
    std::vector<float>& keep_confidences,
    std::vector<Rect2d>& keep_boxes,
    float conf_threshold,
    float iou_threshold,
    const std::string& model_name,
    const int nc
);

std::vector<std::string> classes;


std::string keys =
    "{ help  h     |   | Print help message. }"
    "{ device      | 0 | camera device number. }"
    "{ model       | onnx/models/yolox_s_inf_decoder.onnx | Default model. }"
    "{ yolo        | yolox | yolo model version. }"
    "{ input i     | | Path to input image or video file. Skip this argument to capture frames from a camera. }"
    "{ classes     | | Optional path to a text file with names of classes to label detected objects. }"
    "{ nc          | 80 | Number of classes. Default is 80 (coming from COCO dataset). }"
    "{ thr         | .5 | Confidence threshold. }"
    "{ nms         | .4 | Non-maximum suppression threshold. }"
    "{ mean        | 0.0 0.0 0.0 | Normalization constant. }"
    "{ scale       | 1.0 1.0 1.0 | Preprocess input image by multiplying on a scale factor. }"
    "{ width       | 640 | Preprocess input image by resizing to a specific width. }"
    "{ height      | 640 | Preprocess input image by resizing to a specific height. }"
    "{ rgb         | 1 | Indicate that model works with RGB input images instead BGR ones. }"
    "{ padvalue    | 114.0 | padding value. }"
    "{ paddingmode | 2 | Choose one of computation backends: "
                         "0: resize to required input size without extra processing, "
                         "1: Image will be cropped after resize, "
                         "2: Resize image to the desired size while preserving the aspect ratio of original image }"
    "{ backend     |  0 | Choose one of computation backends: "
                         "0: automatically (by default), "
                         "1: Halide language (http://halide-lang.org/), "
                         "2: Intel's Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit), "
                         "3: OpenCV implementation, "
                         "4: VKCOM, "
                         "5: CUDA }"
    "{ target      | 0 | Choose one of target computation devices: "
                         "0: CPU target (by default), "
                         "1: OpenCL, "
                         "2: OpenCL fp16 (half-float precision), "
                         "3: VPU, "
                         "4: Vulkan, "
                         "6: CUDA, "
                         "7: CUDA fp16 (half-float preprocess) }"
    "{ async       | 0 | Number of asynchronous forwards at the same time. "
                        "Choose 0 for synchronous mode }";

void getClasses(std::string classesFile)
{
    std::ifstream ifs(classesFile.c_str());
    if (!ifs.is_open())
        CV_Error(Error::StsError, "File " + classesFile  + " not found");
    std::string line;
    while (std::getline(ifs, line))
        classes.push_back(line);
}

void drawPrediction(int classId, float conf, int left, int top, int right, int bottom, Mat& frame, Scalar color)
{
    rectangle(frame, Point(left, top), Point(right, bottom), color);

    std::string label = format("%.2f", conf);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ": " + label;
    }

    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - labelSize.height),
              Point(left + labelSize.width, top + baseLine), Scalar::all(255), FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar());
}

void yoloPostProcessing(
    std::vector<Mat>& outs,
    std::vector<int>& keep_classIds,
    std::vector<float>& keep_confidences,
    std::vector<Rect2d>& keep_boxes,
    float conf_threshold,
    float iou_threshold,
    const std::string& model_name,
    const int nc=80)
{
    // Retrieve
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<Rect2d> boxes;

    if (model_name == "yolov8" || model_name == "yolov10" ||
        model_name == "yolov9")
    {
        cv::transposeND(outs[0], {0, 2, 1}, outs[0]);
    }

    if (model_name == "yolonas")
    {
        // outs contains 2 elements of shape [1, 8400, nc] and [1, 8400, 4]. Concat them to get [1, 8400, nc+4]
        Mat concat_out;
        // squeeze the first dimension
        outs[0] = outs[0].reshape(1, outs[0].size[1]);
        outs[1] = outs[1].reshape(1, outs[1].size[1]);
        cv::hconcat(outs[1], outs[0], concat_out);
        outs[0] = concat_out;
        // remove the second element
        outs.pop_back();
        // unsqueeze the first dimension
        outs[0] = outs[0].reshape(0, std::vector<int>{1, outs[0].size[0], outs[0].size[1]});
    }

    // assert if last dim is nc+5 or nc+4
    CV_CheckEQ(outs[0].dims, 3, "Invalid output shape. The shape should be [1, #anchors, nc+5 or nc+4]");
    CV_CheckEQ((outs[0].size[2] == nc + 5 || outs[0].size[2] == nc + 4), true, "Invalid output shape: ");

    for (auto preds : outs)
    {
        preds = preds.reshape(1, preds.size[1]); // [1, 8400, 85] -> [8400, 85]
        for (int i = 0; i < preds.rows; ++i)
        {
            // filter out non object
            float obj_conf = (model_name == "yolov8" || model_name == "yolonas" ||
                              model_name == "yolov9" || model_name == "yolov10") ? 1.0f : preds.at<float>(i, 4) ;
            if (obj_conf < conf_threshold)
                continue;

            Mat scores = preds.row(i).colRange((model_name == "yolov8" || model_name == "yolonas" || model_name == "yolov9" || model_name == "yolov10") ? 4 : 5, preds.cols);
            double conf;
            Point maxLoc;
            minMaxLoc(scores, 0, &conf, 0, &maxLoc);

            conf = (model_name == "yolov8" || model_name == "yolonas" || model_name == "yolov9" || model_name == "yolov10") ? conf : conf * obj_conf;
            if (conf < conf_threshold)
                continue;

            // get bbox coords
            float* det = preds.ptr<float>(i);
            double cx = det[0];
            double cy = det[1];
            double w = det[2];
            double h = det[3];

            // [x1, y1, x2, y2]
            if (model_name == "yolonas" || model_name == "yolov10"){
                boxes.push_back(Rect2d(cx, cy, w, h));
            } else {
                boxes.push_back(Rect2d(cx - 0.5 * w, cy - 0.5 * h,
                                        cx + 0.5 * w, cy + 0.5 * h));
            }
            classIds.push_back(maxLoc.x);
            confidences.push_back(static_cast<float>(conf));
        }
    }

    spdlog::info("Boxes: {}, Confidences: {}", boxes.size(), confidences.size());

    // NMS
    std::vector<int> keep_idx;
    NMSBoxes(boxes, confidences, conf_threshold, iou_threshold, keep_idx);

    for (auto i : keep_idx)
    {
        keep_classIds.push_back(classIds[i]);
        keep_confidences.push_back(confidences[i]);
        keep_boxes.push_back(boxes[i]);
    }
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

/**
 * @function main
 * @brief Main function
 * example call: ./bin/yolo_detector models/V1/trained-ld/17T/17T_weights.onnx models/V1/trained-ld/17T/bounding_boxes.csv tests/sample_data/img/l8_17T_00277.png
 */
int main(int argc, char** argv)
{
    if (argc < 4)
    {
        spdlog::error("Usage: {} <path_to_ld_onnx_file> <path_to_bounding_box.csv> <path_to_sample_image>", argv[0]);
        return 1;
    }
    std::string ld_onnx_file_path = argv[1];
    std::string bounding_box_path = argv[2];
    std::string sample_image_path = argv[3];

    // std::cout << cv::getBuildInformation() << "\n";

    // if model is default, use findFile to get the full path otherwise use the given path
    std::string yolo_model = "yolov8"; // yolo model version
    int nc = 494; // parser.get<int>("nc"); // number of classes. From bounding_boxes.csv

    float confThreshold = 0.5f; // confidence threshold
    float nmsThreshold = 0.45f;
    //![preprocess_params]
    float paddingValue = 0.0f;
    bool swapRB = true; // true = rgb
    int inpWidth = 4608;
    int inpHeight = 4608;
    Scalar scale = Scalar(1.0/255.0,1.0/255.0,1.0/255.0);
    Scalar mean = Scalar(0.0,0.0,0.0);
    ImagePaddingMode paddingMode = static_cast<ImagePaddingMode>(2);
    //![preprocess_params]

    spdlog::info("Parameters initialized");

    // load model
    //![read_net]
    auto start = std::chrono::high_resolution_clock::now();
    Net net = readNet(ld_onnx_file_path);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    spdlog::info("Model loaded from: {} (took {} ms)", ld_onnx_file_path, duration.count());
    
    int backend = 0; // automatically, opencv implementation or cuda?
    int target = 0; // 0: cpu, 6: cuda, 7: cuda fp16
    spdlog::info("Backend and target set");
    //![read_net]

    Mat img = imread(sample_image_path, IMREAD_COLOR);
    spdlog::info("Image loaded: {}", sample_image_path);

    // image pre-processing
    //![preprocess_call]
    Size size(inpWidth, inpHeight);
    Image2BlobParams imgParams(
        scale,
        size,
        mean,
        swapRB,
        CV_32F,
        DNN_LAYOUT_NCHW,
        paddingMode); // , paddingValue);
    spdlog::info("Image2BlobParams created");

    // rescale boxes back to original image
    // Image2BlobParams paramNet;
    // paramNet.scalefactor = scale;
    // paramNet.size = size;
    // paramNet.mean = mean;
    // paramNet.swapRB = swapRB;
    // paramNet.paddingmode = paddingMode;
    //![preprocess_call]

    Mat inp;
    //![preprocess_call_func]
    inp = blobFromImageWithParams(img, imgParams);
    spdlog::info("Blob created from image");
    //![preprocess_call_func]

    //![forward_buffers]
    std::vector<Mat> outs;
    std::vector<int> keep_classIds;
    std::vector<float> keep_confidences;
    std::vector<Rect2d> keep_boxes;
    std::vector<Rect> boxes;
    spdlog::info("Forward buffers initialized");
    //![forward_buffers]

    //![forward]
    start = std::chrono::high_resolution_clock::now();
    net.setInput(inp);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    spdlog::info("Input set to network (took {} ms)", duration.count());

    net.setPreferableBackend(5); // backend);
    net.setPreferableTarget(6); // target);
    start = std::chrono::high_resolution_clock::now();
    net.forward(outs, net.getUnconnectedOutLayersNames());
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    spdlog::info("Forward pass completed, outputs: {}, (took {} ms)", outs.size(), duration.count());
    //![forward]

    //![postprocess]
    start = std::chrono::high_resolution_clock::now();
    yoloPostProcessing(
        outs, keep_classIds, keep_confidences, keep_boxes,
        confThreshold, nmsThreshold,
        yolo_model,
        nc);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    spdlog::info("Post-processing completed, detections: {} (took {} ms)", keep_boxes.size(), duration.count());
    //![postprocess]

    // covert Rect2d to Rect
    //![draw_boxes]
    for (auto box : keep_boxes)
    {
        boxes.push_back(Rect(cvFloor(box.x), cvFloor(box.y), cvFloor(box.width - box.x), cvFloor(box.height - box.y)));
    }
    spdlog::info("Converted {} Rect2d to Rect", boxes.size());

    for (auto& b : boxes) {
        b = scaleBoxBackLetterbox(b, img.size(), size);
    }
    spdlog::info("Boxes scaled back to original image");
    // paramNet.blobRectsToImageRects(boxes, boxes, img.size());

    for (size_t idx = 0; idx < boxes.size(); ++idx)
    {
        Rect box = boxes[idx];
        drawPrediction(keep_classIds[idx], keep_confidences[idx], box.x, box.y,
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
            if (keep_classIds[idx] != class_id) {
                continue;
            }
            Rect pred_box = boxes[idx];
            float iou = (pred_box & gt_boxes.back()).area() / float((pred_box | gt_boxes.back()).area());
            if (iou > 0.5) {
                spdlog::info("True positive: class_id={}, iou={:.2f}", keep_classIds[idx], iou);
            } else {
                spdlog::info("False positive: class_id={}, iou={:.2f}", keep_classIds[idx], iou);
            }
        }
    }
    spdlog::info("Loaded {} ground truth boxes", gt_boxes.size());

    cv::imwrite("debug.jpg", img);
    spdlog::info("Image displayed");
    //![draw_boxes]

    // TODO: evaluate inference error
    // success: IoU > 0.5, failure: IoU <= 0.5

    outs.clear();
    keep_classIds.clear();
    keep_confidences.clear();
    keep_boxes.clear();
    boxes.clear();
    spdlog::info("Buffers cleared, execution complete");
}