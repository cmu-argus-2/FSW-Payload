import torch
import numpy as np
from PIL import Image
import tensorrt as trt
import os
import sys
import torchvision
from ultralytics import YOLO
# from ultralytics.engine.results import Result
import cv2
from dataclasses import dataclass, field
from typing import Dict, List, Sequence
from torchvision.ops import nms
import time
import pycuda.driver as cuda
import pycuda.autoinit
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import onnxruntime as ort
from ultralytics.utils.ops import non_max_suppression # Use ultralytics NMS


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class PTModel:
    def __init__(self, model_path, device="cpu"):
        self.model = YOLO(model_path)
        self.model.to(device)

    def __call__(self, img: Image.Image, conf, imgsz, verbose):
        results = self.model.predict(
            img,
            conf=conf,
            imgsz=imgsz,
            verbose=verbose,
        )
        return results[0]

    def post_process(self, result, imgsz):
        landmarks = result.boxes

        xywh = landmarks.xywh.cpu().numpy()
        class_ids = landmarks.cls.cpu().numpy().astype(int)
        confidences = landmarks.conf.cpu().numpy()

        valid_indices = (
            np.all(xywh >= 0, axis=1)
            & (xywh[:, 0] <= imgsz[0] - 1)
            & (xywh[:, 1] <= imgsz[1] - 1)
        )
        if not np.all(valid_indices):
            if np.any(valid_indices):
                xywh = xywh[valid_indices]
                class_ids = class_ids[valid_indices]
                confidences = confidences[valid_indices]
            else:
                xywh = np.zeros((0, 4))
                class_ids = np.zeros(0, dtype=int)
                confidences = np.zeros(0)
                print("Warning: All detected landmarks have invalid bounding boxes. Returning empty detections.")
        
        return xywh, confidences, class_ids

# available_providers = ort.get_available_providers()
# providers = [p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"] if p in available_providers]
# session = ort.InferenceSession(ONNX_MODEL_PATH, providers=providers or available_providers)
# input_name = session.get_inputs()[0].name
# output_name = session.get_outputs()[0].name
# input_shape = session.get_inputs()[0].shape[2:] # (640, 640)

class ONNXModel:
    def __init__(self,engine_path: str):
        self.engine_path = engine_path
        available_providers = ort.get_available_providers()
        providers = [p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"] if p in available_providers]
        self.session = ort.InferenceSession(engine_path, providers=providers or available_providers)
        print(f"ONNX input name: {self.session.get_inputs()[0].name}")
        print(f"ONNX output name: {self.session.get_outputs()[0].name}")
        print(f"ONNX input shape: {self.session.get_inputs()[0].shape[2:]}")
        # self.onnx_model  = onnx.load(filename)
    
    def __call__(self,x:np.ndarray):
        return self.session.run(None, {"images":  x.astype(np.float32)})[0]

class TrtModel:
    def __init__(self,engine_path,max_batch_size=1,dtype=np.float32):
        
        self.engine_path = engine_path
        self.dtype = dtype
        self.logger = trt.Logger(trt.Logger.VERBOSE)
        self.runtime = trt.Runtime(self.logger)
        
        # Track deserialization time
        deserialize_start = time.time()
        self.engine = self.load_engine(self.runtime, self.engine_path)
        self.deserialization_time = time.time() - deserialize_start
        
        self.max_batch_size = max_batch_size
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
        
        # Track context creation and GPU memory
        context_start = time.time()
        self.context = self.engine.create_execution_context()
        self.context_creation_time = time.time() - context_start
        
        self.gpu_memory_allocated_mb = self.context.engine.device_memory_size / (1024 * 1024)
        

    @staticmethod
    def load_engine(trt_runtime, engine_path):
        trt.init_libnvinfer_plugins(None, "")             
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        engine = trt_runtime.deserialize_cuda_engine(engine_data)
        return engine
    
    def allocate_buffers(self):
        
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        
        for binding in self.engine:
            size = trt.volume(self.engine.get_tensor_shape(binding)) * self.max_batch_size
            host_mem = cuda.pagelocked_empty(size, self.dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            bindings.append(int(device_mem))

            if self.engine.get_tensor_mode(binding)==trt.TensorIOMode.INPUT:
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        
        return inputs, outputs, bindings, stream
       
    
    def __call__(self,x:np.ndarray,batch_size=2):
        
        x = x.astype(self.dtype)
        
        np.copyto(self.inputs[0].host,x.ravel())
        
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp.device, inp.host, self.stream)
        
        self.context.execute_async(batch_size=batch_size, bindings=self.bindings, stream_handle=self.stream.handle)
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream) 
            
        
        self.stream.synchronize()
        for binding in self.engine:
            if self.engine.get_tensor_mode(binding)==trt.TensorIOMode.OUTPUT:
                output_shape = self.engine.get_tensor_shape(binding)
                return [out.host.reshape(output_shape) for out in self.outputs]
        return [out.host.reshape(batch_size,-1) for out in self.outputs]


def preprocess_image(img_array, target_size=4608): # input image will be at most 4608x2592
    height, width = img_array.shape[:2]
    
    if isinstance(target_size,int):
        target_width = target_size
        target_height = target_size
    elif len(target_size) == 2:
        target_width = target_size[1]
        target_height = target_size[0]
    else:
        print("invalid target size")
        return
    
    # Resize to different target size while maintaining aspect ratio
    height_scale = target_height / height
    width_scale  = target_width / width
    scale = min(height_scale, width_scale)
    new_width = int(width * scale)
    new_height = int(height * scale)
    img_array = cv2.resize(img_array, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # Letterbox to 4608x4608
    pad_top = np.max((target_height - new_height) // 2,0)
    pad_bottom = target_height - new_height - pad_top
    pad_left = np.max((target_width - new_width) // 2,0)
    pad_right = target_width - new_width - pad_left
    img_letterboxed = np.pad(img_array, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant', constant_values=0)

    # img = Image.fromarray(img_letterboxed)

    img_letterboxed = np.expand_dims(img_letterboxed, axis=0)
    # NCHW
    img_letterboxed = np.transpose(img_letterboxed, (0, 3, 1, 2)) / 255.0
    
    return img_letterboxed

def xywh_to_xyxy(boxes):
    x, y, w, h = boxes.unbind(-1)
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return torch.stack((x1, y1, x2, y2), dim=-1)

def xyxy_to_xywh(boxes):
    x1, y1, x2, y2 = boxes.unbind(-1)
    x = (x1 + x2) / 2
    y = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack((x, y, w, h), dim=-1)

def yolo_postprocess(pred: torch.Tensor, conf_thres=0.5, iou_thres=0.45):
    """
    pred: tensor of shape (1, 4 + nc, nb)
    returns: boxes, scores, class_ids
    """
    pred = pred.transpose(0, 1)   # (nb, 4 + nc)

    boxes = pred[:, :4]                      # (nb, 4)
    cls_scores = pred[:, 4:]                 # (nb, nc)

    scores, class_ids = cls_scores.max(dim=1)

    keep = scores > conf_thres
    boxes = xywh_to_xyxy(boxes[keep])
    
    scores = scores[keep]
    class_ids = class_ids[keep]

    # boxes must be in xyxy format for torchvision NMS
    keep_idx = torchvision.ops.batched_nms(boxes, scores, class_ids, iou_thres)

    return xyxy_to_xywh(boxes[keep_idx]), scores[keep_idx], class_ids[keep_idx]


def _aligned_table(headers, rows):
    """Return a list of lines forming a padded markdown table."""
    widths = [len(h) for h in headers]
    str_rows = []
    for row in rows:
        str_row = [str(v) for v in row]
        str_rows.append(str_row)
        for i, cell in enumerate(str_row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(cells):
        return "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells)) + " |"

    lines = [
        fmt_row(headers),
        "| " + " | ".join("-" * w for w in widths) + " |",
    ]
    for row in str_rows:
        lines.append(fmt_row(row))
    return lines


def _build_detection_rows(sorted_class_ids, sorted_confidences, sorted_boxes,
                          true_class_ids, true_labels):
    """
    For each prediction produce a row:
      (class_id, confidence, iou, center_dist_x, center_dist_y, TP?)
    sorted_boxes are in (cx, cy, w, h) pixel space.
    true_labels rows are [class_id, cx_px, cy_px, w_px, h_px].
    """
    rows = []
    for i, cls in enumerate(sorted_class_ids):
        conf = float(sorted_confidences[i])
        box  = sorted_boxes[i]
        box_t = torch.tensor(box[:4]) if isinstance(box, np.ndarray) else box[:4].float()

        gt_idx = np.where(true_class_ids == int(cls))[0]
        if len(gt_idx):
            gt = true_labels[gt_idx[0]]
            gt_t  = torch.from_numpy(gt[1:5]).float()
            iou   = torchvision.ops.box_iou(
                xywh_to_xyxy(gt_t.unsqueeze(0)),
                xywh_to_xyxy(box_t.unsqueeze(0))
            ).item()
            pred_c = (box_t[:2]).numpy() if isinstance(box_t, torch.Tensor) else box_t[:2]
            gt_c   = gt[1:3]
            cdx, cdy = float(pred_c[0] - gt_c[0]), float(pred_c[1] - gt_c[1])
            iou_s  = f"{iou:.3f}"
            cdx_s  = f"{cdx:+.1f}"
            cdy_s  = f"{cdy:+.1f}"
            is_tp  = iou > 0.5
        else:
            iou_s = cdy_s = cdx_s = "N/A"
            is_tp = False

        rows.append((int(cls), f"{conf:.3f}", iou_s, cdx_s, cdy_s, "yes" if is_tp else "no"))
    return rows


def _write_model_section(f, label, exists, sorted_class_ids, sorted_confidences,
                         sorted_boxes, true_class_ids, true_labels,
                         inference_time, postprocess_time,
                         extra_lines=None):
    """Write one model's ## section to file f. extra_lines is a list of
    extra '- key: value' strings inserted after timing (e.g. memory, deserialization)."""
    if not exists:
        return

    rows = _build_detection_rows(sorted_class_ids, sorted_confidences, sorted_boxes,
                                 true_class_ids, true_labels)
    tp = sum(1 for r in rows if r[-1] == "yes")
    fp = sum(1 for r in rows if r[-1] == "no" and r[2] != "N/A")  # detected, GT exists, IoU<=0.5
    fp_no_gt = sum(1 for r in rows if r[2] == "N/A")              # detected, no GT class
    fn = sum(1 for cls in true_class_ids if cls not in [int(r[0]) for r in rows])
    total_gt = len(true_class_ids)
    recall    = tp / total_gt if total_gt > 0 else 0.0
    precision = tp / len(rows) if len(rows) > 0 else 0.0
    status = "PASS" if tp > 0 else "FAIL"

    f.write(f"\n### [{status}] {label}\n\n")
    f.write(f"- Inference time:     {inference_time:.4f} s\n")
    f.write(f"- Post-process time:  {postprocess_time:.4f} s\n")
    if extra_lines:
        for line in extra_lines:
            f.write(f"- {line}\n")
    f.write(f"- Ground-truth boxes: {total_gt}\n")
    f.write(f"- Predictions:        {len(rows)}\n")
    f.write(f"- TP: {tp} | FP (low IoU): {fp} | FP (no GT): {fp_no_gt} | FN: {fn}\n")
    f.write(f"- Recall: {recall:.4f} | Precision: {precision:.4f}\n")

    if rows:
        f.write("\n")
        headers = ["class_id", "confidence", "iou", "center_dx", "center_dy", "TP?"]
        for line in _aligned_table(headers, rows):
            f.write(line + "\n")
    f.write("\n")


def run_single(region_id, image_id, model_version, tgt_imgsz, fpstring, nms_string, pngstring, use_jpg, results_folder):
    pt_model_path    = f"models/{model_version}/trained-ld/{region_id}/{region_id}_weights.pt"
    trt_engine_path  = f"models/{model_version}/trained-ld/{region_id}/{region_id}_weights_{fpstring}_sz_{tgt_imgsz[1]}{nms_string}.trt"
    onnx_engine_path = f"models/{model_version}/trained-ld/{region_id}/{region_id}_weights_{fpstring}_sz_{tgt_imgsz[1]}.onnx"
    bbox_path = f"models/{model_version}/trained-ld/{region_id}/bounding_boxes.csv"
    image_path = f"models/{model_version}/sample_images/l8_{region_id}_{image_id}.{pngstring}"
    label_path = f"models/{model_version}/sample_images/l8_{region_id}_{image_id}.txt"

    if not os.path.exists(image_path):
        if use_jpg:
            png_image_path = image_path.replace("jpg","png")
            if not os.path.exists(png_image_path):
                print(f"Image not found: {image_path}")
                return
            else:
                # convert png to jpg
                img_png = Image.open(png_image_path)
                img_png.save(image_path)
        else:
            print(f"Image not found: {image_path}")
            return
    
    start_time = time.time()
    img = Image.open(image_path).convert("RGB")
    img_loading_time = time.time() - start_time
    
    # Crop image to 4608x2592 and letterbox to 4608x4608
    img_array = np.array(img)
    height, width = img_array.shape[:2]
    
    labels = np.loadtxt(label_path, ndmin=2)
    
    # Crop to max 4608 by 2592 while maintaining center
    IMAGE_WIDTH = 4608
    IMAGE_HEIGHT = 2592
    if height > IMAGE_HEIGHT:
        top = (height - IMAGE_HEIGHT) // 2
        img_array = img_array[top:top + IMAGE_HEIGHT,:]
    
    if width > IMAGE_WIDTH:
        left = (width - IMAGE_WIDTH) // 2
        img_array = img_array[:, left:left + IMAGE_WIDTH]

    height, width = img_array.shape[:2] # should be 4608x2592 after cropping
    batch_size = 1
    
    trt_engine_exists  = os.path.exists(trt_engine_path)
    pt_engine_exists   = os.path.exists(pt_model_path)
    onnx_engine_exists = os.path.exists(onnx_engine_path)
    
    # Original Pytorch model inference for comparison
    # torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pt_engine_exists:
        pt_model  = PTModel(pt_model_path, "cuda" if torch.cuda.is_available() else "cpu")
        start_time = time.time()
        result = pt_model(img, 0.5, (2592,4608), True)
        pt_inference_time = time.time() - start_time
        
        start_time = time.time()
        pt_xywh, pt_confidences, pt_class_ids = pt_model.post_process(result, (IMAGE_WIDTH, IMAGE_HEIGHT))
        pt_postprocess_time = time.time() - start_time

        # Release PyTorch's CUDA memory cache before TensorRT allocates its workspace.
        # PyTorch's caching allocator holds onto GPU memory even after tensors are freed,
        # which fragments the heap and forces TRT into slow allocation paths.
        del pt_model
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    else:
        print(f"{pt_model_path} not found")

    if trt_engine_exists:
        trt_model = TrtModel(trt_engine_path)
        
        img_letterboxed = preprocess_image(img_array, target_size=tgt_imgsz)
        
        start_time = time.time()
        result = trt_model(img_letterboxed, batch_size)
        trt_inference_time = time.time() - start_time
        
        start_time = time.time()
        result_array = result[0].squeeze()
        print(f"TensorRT output shape: {result_array.shape}")
        trt_boxes, trt_confidences, trt_class_ids = yolo_postprocess(torch.from_numpy(result_array), conf_thres=0.5, iou_thres=0.45)
        # TODO: Make this general
        # trt_boxes[:, 1] -= 1008
        trt_postprocess_time = time.time() - start_time
    else:
        print(f"{trt_engine_path} not found")

    # ONNX runtime
    if onnx_engine_exists:
        onnx_model = ONNXModel(onnx_engine_path)
        
        img_letterboxed = preprocess_image(img_array, target_size=4608)

        start_time = time.time()
        onnx_results = onnx_model(img_letterboxed)
        onnx_inference_time = time.time() - start_time
        start_time = time.time()
        onnx_boxes, onnx_confidences, onnx_class_ids = yolo_postprocess(torch.from_numpy(onnx_results.squeeze()), conf_thres=0.5, iou_thres=0.45)
        onnx_boxes[:, 1] -= 1008
        onnx_postprocess_time = time.time() - start_time
    else:
        print(f"{onnx_engine_path} not found")

    bboxes = np.loadtxt(bbox_path, delimiter=",", skiprows=1)
    
    # from labels
    # true_sorted_class_ids
    
    # compare pth and trt results
    # Sort both by class_id
    if pt_engine_exists:
        pt_sort_idx = np.argsort(pt_class_ids)
        pt_sorted_class_ids = pt_class_ids[pt_sort_idx]
        pt_sorted_confidences = pt_confidences[pt_sort_idx]
        pt_sorted_boxes = pt_xywh[pt_sort_idx,:]
    
    if trt_engine_exists:
        trt_sort_idx = np.argsort(trt_class_ids)
        trt_sorted_class_ids = trt_class_ids[trt_sort_idx]
        trt_sorted_confidences = trt_confidences[trt_sort_idx]
        trt_sorted_boxes = trt_boxes[trt_sort_idx]
    
    if onnx_engine_exists:
        onnx_sort_idx = np.argsort(onnx_class_ids)
        onnx_sorted_class_ids = onnx_class_ids[onnx_sort_idx]
        onnx_sorted_confidences = onnx_confidences[onnx_sort_idx]
        onnx_sorted_boxes = onnx_boxes[onnx_sort_idx]
    
    # already sorted
    true_class_ids = labels[:, 0].astype(int)
    true_labels = labels * np.array([1,width, height, width, height])  # Scale normalized coordinates to pixel values
    # trt_labels = np.array([0,0, 1008, 0, 0]) + labels * np.array([1, width, height, width, height])  # Scale normalized coordinates to pixel values
    
    max_len = len(true_class_ids)
    all_class_ids = set(true_class_ids)
    if pt_engine_exists:
        max_len = max(max_len, len(pt_sorted_class_ids))
        all_class_ids = all_class_ids | set(pt_sorted_class_ids)
    if trt_engine_exists:
        max_len = max(max_len, len(trt_sorted_class_ids))
        all_class_ids = all_class_ids | set(int(x) for x in trt_sorted_class_ids)
    if onnx_engine_exists:
        max_len = max(max_len, len(onnx_sorted_class_ids))
        all_class_ids = all_class_ids | set(int(x) for x in onnx_sorted_class_ids)
    
    all_class_ids = list(all_class_ids)
    all_class_ids.sort()
    max_len = len(all_class_ids)

    output_txt_file = os.path.join(results_folder, f"detection_comparison_{region_id}_{image_id}.txt")
    with open(output_txt_file, "w") as f:
        f.write(f"# LD Comparison: Region {region_id} — Image {image_id}\n\n")
        f.write(f"**Config:** {fpstring}, sz={tgt_imgsz}, {pngstring}\n\n")
        f.write("---\n\n")
        f.write("## Performance\n\n")
        f.write(f"- Image loading time: {img_loading_time:.4f} s\n")
        if trt_engine_exists:
            f.write(f"- TRT deserialization:   {trt_model.deserialization_time:.4f} s\n")
            f.write(f"- TRT context creation:  {trt_model.context_creation_time:.4f} s\n")
            f.write(f"- TRT GPU memory:        {trt_model.gpu_memory_allocated_mb:.1f} mb\n")
        f.write("\n---\n")

        if trt_engine_exists:
            _write_model_section(
                f, "TensorRT", trt_engine_exists,
                trt_sorted_class_ids.numpy() if isinstance(trt_sorted_class_ids, torch.Tensor) else np.array([int(x) for x in trt_sorted_class_ids]),
                trt_sorted_confidences.numpy() if isinstance(trt_sorted_confidences, torch.Tensor) else np.array(trt_sorted_confidences),
                trt_sorted_boxes,
                true_class_ids, true_labels,
                trt_inference_time, trt_postprocess_time,
            )

        if pt_engine_exists:
            _write_model_section(
                f, "PyTorch", pt_engine_exists,
                pt_sorted_class_ids, pt_sorted_confidences, pt_sorted_boxes,
                true_class_ids, true_labels,
                pt_inference_time, pt_postprocess_time,
            )

        if onnx_engine_exists:
            _write_model_section(
                f, "ONNX", onnx_engine_exists,
                onnx_sorted_class_ids.numpy() if isinstance(onnx_sorted_class_ids, torch.Tensor) else np.array([int(x) for x in onnx_sorted_class_ids]),
                onnx_sorted_confidences.numpy() if isinstance(onnx_sorted_confidences, torch.Tensor) else np.array(onnx_sorted_confidences),
                onnx_sorted_boxes,
                true_class_ids, true_labels,
                onnx_inference_time, onnx_postprocess_time,
            )

    print(f"Report written to: {output_txt_file}")
    
    # Plot the results of both compared to the real boxes
    try:
        if pt_engine_exists:
            fig, ax = plt.subplots()
            ax.imshow(img)
            for i, true_label in enumerate(true_labels):
                # Ground truth
                rect = patches.Rectangle((true_label[1], true_label[2]), true_label[3], true_label[4], linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                # PyTorch detections
            for i in range(pt_sorted_boxes.shape[0]):
                rect = patches.Rectangle((pt_sorted_boxes[i, 0], pt_sorted_boxes[i, 1]), pt_sorted_boxes[i, 2], pt_sorted_boxes[i, 3], linewidth=1, edgecolor='b', facecolor='none')
                ax.add_patch(rect)
            fig.savefig(results_folder + f"comparison_plot_pt_{region_id}_{image_id}.png")
        if trt_engine_exists:
            fig2, ax2 = plt.subplots()
            ax2.imshow(img)
            # ax2.imshow(img_letterboxed[0].transpose(1,2,0))
            for i, true_label in enumerate(true_labels):
                # Ground truth
                rect = patches.Rectangle((true_label[1], true_label[2]), true_label[3], true_label[4], linewidth=1, edgecolor='r', facecolor='none')
                ax2.add_patch(rect)
                # PyTorch detections
            for i in range(len(trt_sorted_boxes)):
                rect = patches.Rectangle((trt_sorted_boxes[i, 0], trt_sorted_boxes[i, 1]), trt_sorted_boxes[i, 2], trt_sorted_boxes[i, 3], linewidth=1, edgecolor='g', facecolor='none')
                ax2.add_patch(rect)
            fig2.savefig(results_folder + f"comparison_plot_trt_{region_id}_{image_id}.png")
        if onnx_engine_exists:
            fig3, ax3 = plt.subplots()
            ax3.imshow(img)
            # ax2.imshow(img_letterboxed[0].transpose(1,2,0))
            for i, true_label in enumerate(true_labels):
                # Ground truth
                rect = patches.Rectangle((true_label[1], true_label[2]), true_label[3], true_label[4], linewidth=1, edgecolor='r', facecolor='none')
                ax3.add_patch(rect)
                # PyTorch detections
            for i in range(len(onnx_sorted_boxes)):
                rect = patches.Rectangle((onnx_sorted_boxes[i, 0], onnx_sorted_boxes[i, 1]), onnx_sorted_boxes[i, 2], onnx_sorted_boxes[i, 3], linewidth=1, edgecolor='g', facecolor='none')
                ax3.add_patch(rect)
            fig3.savefig(results_folder + f"comparison_plot_onnx_{region_id}_{image_id}.png")
    except Exception as e:
        print(f"Error occurred while saving plots: {e}")


if __name__ == "__main__":
    model_version = "V1"
    tgt_imgsz     = (2592, 4608)  # (H, W)
    fp16          = True
    trt_with_nms  = False
    use_jpg       = False

    fpstring   = "fp16" if fp16 else "fp32"
    nms_string = "_nms" if trt_with_nms else ""
    pngstring  = "jpg"  if use_jpg else "png"

    # ── Add / remove entries here to control which images are processed ──
    samples = [
        ("17T", "00330"),
        ("17R", "00168"),
    ]

    script_dir     = os.path.dirname(os.path.abspath(__file__))
    results_folder = os.path.join(script_dir, f"../../results/ld_comp_{fpstring}_sz_{tgt_imgsz}_{pngstring}/")
    os.makedirs(results_folder, exist_ok=True)

    for region_id, image_id in samples:
        print(f"\n{'='*60}")
        print(f"Running  region={region_id}  image={image_id}")
        print(f"{'='*60}")
        run_single(region_id, image_id, model_version, tgt_imgsz,
                   fpstring, nms_string, pngstring, use_jpg, results_folder)