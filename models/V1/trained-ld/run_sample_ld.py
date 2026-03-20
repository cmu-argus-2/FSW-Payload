import torch
import numpy as np
from PIL import Image
import tensorrt as trt
import os
import torchvision
from ultralytics import YOLO
# from ultralytics.engine.results import Result
import cv2
from dataclasses import dataclass, field
from typing import Dict, List, Sequence
from torchvision.ops import nms
import pycuda.driver as cuda
import pycuda.autoinit
import matplotlib.pyplot as plt
import matplotlib.patches as patches



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


class TrtModel:
    def __init__(self,engine_path,max_batch_size=1,dtype=np.float32):
        
        self.engine_path = engine_path
        self.dtype = dtype
        self.logger = trt.Logger(trt.Logger.VERBOSE)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.load_engine(self.runtime, self.engine_path)
        self.max_batch_size = max_batch_size
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
        self.context = self.engine.create_execution_context()

    @staticmethod
    def load_engine(trt_runtime, engine_path):
        trt.init_libnvinfer_plugins(None, "")             
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        engine = trt_runtime.deserialize_cuda_engine(engine_data)
        return engine

    def preprocess_image(self,img_array, target_size=4608): # input image will be at most 4608x2592
        height, width = img_array.shape[:2]
        
        # Resize to different target size while maintaining aspect ratio
        height_scale = target_size / height
        width_scale  = target_size / width
        scale = min(height_scale, width_scale)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img_array = cv2.resize(img_array, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        # Letterbox to 4608x4608
        pad_top = np.max((target_size - height) // 2,0)
        pad_bottom = target_size - height - pad_top
        pad_left = np.max((target_size - width) // 2,0)
        pad_right = target_size - width - pad_left
        img_letterboxed = np.pad(img_array, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant', constant_values=0)

        # img = Image.fromarray(img_letterboxed)

        img_letterboxed = np.expand_dims(img_letterboxed, axis=0)
        # NCHW
        img_letterboxed = np.transpose(img_letterboxed, (0, 3, 1, 2)) / 255.0
        
        return img_letterboxed
    
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


if __name__ == "__main__":
    # Paths
    model_version = "V1"
    region_id = "17R"
    image_id = "00040"
    pt_model_path = f"models/{model_version}/trained-ld/{region_id}/{region_id}_weights.pt"
    trt_engine_path = f"models/{model_version}/trained-ld/{region_id}/{region_id}_weights.trt"
    bbox_path = f"models/{model_version}/trained-ld/{region_id}/bounding_boxes.csv"
    image_path = f"models/{model_version}/sample_images/l8_{region_id}_{image_id}.png"
    label_path = f"models/{model_version}/sample_images/l8_{region_id}_{image_id}.txt"
    
    img = Image.open(image_path).convert("RGB")
    
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
    
    model = TrtModel(trt_engine_path)
    batch_size = 1
    img_letterboxed = model.preprocess_image(img_array)
    
    result = model(img_letterboxed,batch_size)
    result_array = result[0].squeeze()
    print(f"TensorRT output shape: {result_array.shape}")
    trt_boxes, trt_confidences, trt_class_ids = yolo_postprocess(torch.from_numpy(result_array), conf_thres=0.5, iou_thres=0.45)
    # TODO: Make this general
    trt_boxes[:, 1] -= 1008
    
    # Original Pytorch model inference for comparison
    # torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = PTModel(pt_model_path, device="cpu")
    result = model(img, 0.5, 4608, True)
    pt_xywh, pt_confidences, pt_class_ids = model.post_process(result, (IMAGE_WIDTH, IMAGE_HEIGHT))
    
    print(f"Input tensor shape: {img.size}")
    # print(result)

    bboxes = np.loadtxt(bbox_path, delimiter=",", skiprows=1)
    
    # from labels
    # true_sorted_class_ids
    
    # compare pth and trt results
    # Sort both by class_id
    pt_sort_idx = np.argsort(pt_class_ids)
    pt_sorted_class_ids = pt_class_ids[pt_sort_idx]
    pt_sorted_confidences = pt_confidences[pt_sort_idx]
    pt_sorted_boxes = pt_xywh[pt_sort_idx,:]
    
    trt_sort_idx = np.argsort(trt_class_ids)
    trt_sorted_class_ids = trt_class_ids[trt_sort_idx]
    trt_sorted_confidences = trt_confidences[trt_sort_idx]
    trt_sorted_boxes = trt_boxes[trt_sort_idx]
    
    # already sorted
    true_class_ids = labels[:, 0].astype(int)
    true_labels = labels * np.array([1,width, height, width, height])  # Scale normalized coordinates to pixel values
    # trt_labels = np.array([0,0, 1008, 0, 0]) + labels * np.array([1, width, height, width, height])  # Scale normalized coordinates to pixel values
    
    
    max_len = max(len(pt_sorted_class_ids), len(trt_sorted_class_ids), len(true_class_ids))
    
    all_class_ids = list(set(pt_sorted_class_ids) | set(int(x) for x in trt_sorted_class_ids) | set(true_class_ids))
    all_class_ids.sort()
    max_len = len(all_class_ids)
    results_folder = f"models/{model_version}/results/ld_comp_{region_id}_{image_id}"
    os.makedirs(results_folder, exist_ok=True)
    
    # TODO: Make these consistent. Show the detections in an ordered way
    # show the same attributes for both pytorch and tensorrt
    # TODO: Compute the accuracy metrics
    # TP = IOU > 0.5 with correct class_id
    # recall = TP / (TP + FN)
    # precision = TP / (TP + FP)
    print("PyTorch vs TensorRT detections (sorted by class_id):")
    is_landmark = False
    pt_results = {"fp_count": 0, "fn_count": 0, "tp_count": 0, "tn_count": 0}
    trt_results = {"fp_count": 0, "fn_count": 0, "tp_count": 0, "tn_count": 0}
    for i, class_id in enumerate(all_class_ids):
        if class_id in true_class_ids:
            idx = np.where(true_class_ids == class_id)[0][0]
            print(f"Ground Truth [{i}]: Class ID: {true_class_ids[idx]}, Pixel Coords: {true_labels[idx, 1:5]}")
            is_landmark = True
        else:
            print(f"Ground Truth [{i}]: No landmark for Class ID: {class_id}")
            is_landmark = False
            
        if class_id in pt_sorted_class_ids:
            idx_pt = np.where(pt_sorted_class_ids == class_id)[0][0]
            # TODO: if is_landmark -> TP OR FP
            # iou, center distance
            if is_landmark:
                pt_results["tp_count"] += 1
                pt_iou = torchvision.ops.box_iou(torch.from_numpy(true_labels[idx, 1:5]), torch.from_numpy(pt_sorted_boxes[idx_pt, :4]))
                pt_cdist = pt_sorted_boxes[idx_pt, :2] - true_labels[idx, 1:3]
                print(f"PyTorch [{i}]: Class ID: {pt_sorted_class_ids[idx_pt]}, Confidence: {pt_sorted_confidences[idx_pt]:.4f}, Box: {pt_sorted_boxes[idx_pt]}, IOU: {pt_iou:.4f}, Center Distance: {pt_cdist}")
            else:
                pt_results["fp_count"] += 1
                print(f"PyTorch [{i}]: Class ID: {pt_sorted_class_ids[idx_pt]}, Confidence: {pt_sorted_confidences[idx_pt]:.4f}, Box: {pt_sorted_boxes[idx_pt]}")
        else:
            print(f"PyTorch [{i}]: No detection for Class ID: {class_id}")
            if is_landmark:
                pt_results["fn_count"] += 1
            else:
                pt_results["tn_count"] += 1
        
        if class_id in trt_sorted_class_ids:
            idx_trt = np.where(trt_sorted_class_ids == class_id)[0][0]
            print(f"TensorRT [{i}]: Class ID: {trt_sorted_class_ids[idx_trt]}, Confidence: {trt_sorted_confidences[idx_trt]:.4f}, Box: {trt_sorted_boxes[idx_trt].tolist():.1f}")
        else:
            print(f"TensorRT [{i}]: No detection for Class ID: {class_id}")
        print()
    
    # Plot the results of both compared to the real boxes
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
    fig.savefig(f"comparison_plot_pt.png")
    
    fig2, ax2 = plt.subplots()
    ax2.imshow(img)
    # ax2.imshow(img_letterboxed[0].transpose(1,2,0))
    for i, true_label in enumerate(true_labels):
        # Ground truth
        rect = patches.Rectangle((true_label[1], true_label[2]), true_label[3], true_label[4], linewidth=1, edgecolor='r', facecolor='none')
        ax2.add_patch(rect)
        # PyTorch detections
    for i in range(len(trt_sorted_boxes)):
        rect = patches.Rectangle((trt_sorted_boxes[i, 0], trt_sorted_boxes[i, 1] - 1008.0), trt_sorted_boxes[i, 2], trt_sorted_boxes[i, 3], linewidth=1, edgecolor='g', facecolor='none')
        ax2.add_patch(rect)
    fig2.savefig(f"comparison_plot_trt.png")
    # TODO: Store the accuracy metrics and the images in the results directory with a consistent naming scheme for later analysis
