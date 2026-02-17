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


@dataclass
class Frame:

    image: np.ndarray
    
    def __init__(self, image_path: str):
        self.image = cv2.imread(image_path)
    
    def resize(self, width: int = 640, height: int = 480) -> np.ndarray:
        """
        Resize the image contained in this Frame to the specified width and height.
        The resized image is returned but this Frame object is not modified.

        :param width: The width to resize the image to.
        :param height: The height to resize the image to.
        :return: The resized image as a numpy array.
        """
        return cv2.resize(self.image, (width, height), interpolation=cv2.INTER_AREA)



@dataclass
class LandmarkDetections:
    """
    A class to store info about landmark detections.

    Attributes:
        pixel_coordinates: A numpy array of shape (N, 2) containing the x and y pixel coordinates
                           for each detected landmark's centroid.
        latlons: A numpy array of shape (N, 2) containing the latitudes and longitudes
                 for each detected landmark's centroid.
        class_ids: A numpy array of shape (N,) containing the class IDs for each detected landmark.
        confidences: A numpy array of shape (N,) containing the confidence scores for each detected landmark.
    """

    pixel_coordinates: np.ndarray
    latlons: np.ndarray
    class_ids: np.ndarray
    region_ids: np.ndarray
    confidences: np.ndarray

    def __len__(self) -> int:
        """
        :return: The number of landmark detections.
        """
        return len(self.class_ids)

    def __getitem__(self, index: int | slice | Sequence[int] | np.ndarray) -> "LandmarkDetections":
        """
        Get a subset of the landmark detections from this LandmarkDetections object.

        Args:
            index: The index of the landmark detections to retrieve.

        Returns:
            A LandmarkDetections object containing the specified entries.
        """
        return LandmarkDetections(
            pixel_coordinates=self.pixel_coordinates[index, :],
            latlons=self.latlons[index, :],
            class_ids=self.class_ids[index],
            region_ids=self.region_ids[index],
            confidences=self.confidences[index],
        )

    def __iter__(self):
        """
        :return: A generator that yields Tuples containing the pixel_coordinates, latlon, class_id, and confidence for each landmark.
        """
        for i in range(len(self)):
            yield (
                self.pixel_coordinates[i, :],
                self.latlons[i, :],
                self.class_ids[i],
                self.region_ids[i],
                self.confidences[i],
            )

    @staticmethod
    def empty() -> "LandmarkDetections":
        """
        Creates an empty LandmarkDetections object.

        Returns:
            A LandmarkDetections object with empty arrays of the correct shape for all attributes.
        """
        return LandmarkDetections(
            pixel_coordinates=np.zeros((0, 2)),
            latlons=np.zeros((0, 2)),
            class_ids=np.zeros(0, dtype=int),
            region_ids=np.array([], dtype="U32"),
            confidences=np.zeros(0),
        )

    @staticmethod
    def stack(detections: List["LandmarkDetections"]) -> "LandmarkDetections":
        """
        Stack multiple LandmarkDetections into a single LandmarkDetections object.

        Args:
            detections: A list of LandmarkDetections objects.

        Returns:
            A LandmarkDetections object containing the stacked data.
        """
        if len(detections) == 0:
            return LandmarkDetections.empty()

        return LandmarkDetections(
            pixel_coordinates=np.row_stack([det.pixel_coordinates for det in detections]),
            latlons=np.row_stack([det.latlons for det in detections]),
            class_ids=np.concatenate([det.class_ids for det in detections]),
            region_ids=np.concatenate([det.region_ids for det in detections]),
            confidences=np.concatenate([det.confidences for det in detections]),
        )



class LandmarkDetector:
    CONFIDENCE_THRESHOLD = 0.5
    IMAGE_SIZE = (2592, 4608)
    def __init__(self, model_weights_path, bbx_path, region_id):
        self.region_id = region_id
        self.model = YOLO(
            model_weights_path
        )
        self.ground_truth = LandmarkDetector.load_ground_truth(
            bbx_path
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    @staticmethod
    def load_ground_truth(ground_truth_path: str) -> np.ndarray:
        """
        Loads ground truth bounding box coordinates from a CSV file.

        Args:
            ground_truth_path (str): Path to the ground truth CSV file.

        Returns:
            A numpy array of shape (N, 6) containing the following for each landmark:
            (centroid_lat, centroid_lon, top_left_lat, top_left_lon, bottom_right_lat, bottom_right_lon).
        """
        return np.loadtxt(ground_truth_path, delimiter=",", skiprows=1)

    def detect_landmarks(
        self, frame: Frame
    ) -> List[LandmarkDetections]:
        """
        Detects landmarks in a set of input images using a pretrained YOLO model and extracts relevant information.

        The detection process filters out landmarks with low confidence scores (below 0.5)
        and invalid bounding box dimensions.
        It aims to provide a comprehensive set of data for each detected landmark,
        facilitating further analysis or processing.

        Args:
            frames: The input Frames on which to perform landmark detection. Does not need to be a multiple of
                    batch_size.
            batch_size: The number of input Frames to process in each batch.

        Returns:
            A LandmarkDetections object containing the detected landmarks and associated data.
        """
        landmark_detections = []
        
        frame = frame.resize(640,480)
        
        results: Result = self.model.predict(
            Image.fromarray(frame.image),
            conf=LandmarkDetector.CONFIDENCE_THRESHOLD,
            imgsz=LandmarkDetector.IMAGE_SIZE,
            verbose=True,
        )

        return results

def load_pytorch_model(model_path):
    """Load a PyTorch model from .pt file"""
    model = torch.load(model_path)
    model.eval()
    return model

def verify_tensorrt_model(engine_path):
    if os.path.exists(engine_path):
        size = os.path.getsize(engine_path)
        print(f"File exists: {engine_path}, Size: {size} bytes")
        
        # Check first few bytes (TensorRT engines should have specific magic bytes)
        with open(engine_path, 'rb') as f:
            header = f.read(16)
            print(f"File header (hex): {header.hex()}")
    else:
        print(f"File does not exist: {engine_path}")


def load_tensorrt_model(engine_path):
    """Load a TensorRT engine from .trt file"""
    logger = trt.Logger(trt.Logger.VERBOSE)
    with open(engine_path, 'rb') as f:
        serialized_engine = f.read()
        print(f"Loaded TensorRT engine from {engine_path}, size: {len(serialized_engine)} bytes")
    with trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        if engine is None:
            print(f"TensorRT version: {trt.__version__}")
            verify_tensorrt_model(engine_path)
            raise RuntimeError(f"Failed to deserialize TensorRT engine from {engine_path}")
    print(f"Successfully deserialized TensorRT engine from {engine_path}")
    return engine

def preprocess_image(image_path, input_size=(640, 480)):
    """Load and preprocess image"""
    img = Image.open(image_path).convert('RGB')
    img = img.resize(input_size)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    return img_tensor, img_array

def run_pytorch_inference(model, image_tensor):
    """Run inference with PyTorch model"""
    with torch.no_grad():
        output = model(image_tensor)
    return output.cpu().numpy()

def run_tensorrt_inference(engine, image_array):
    """Run inference with TensorRT engine"""
    context = engine.create_execution_context()
    
    input_data = np.ascontiguousarray(image_array.flatten(), dtype=np.float32)
    output_data = np.empty(context.get_binding_shape(1), dtype=np.float32)
    # cpu
    # bindings = [int(input_data.ctypes.data), int(output_data.ctypes.data)]
    
    # Allocate device memory
    d_input = cuda.mem_alloc(input_data.nbytes)
    d_output = cuda.mem_alloc(output_data.nbytes)

    # Copy input to device
    cuda.memcpy_htod(d_input, input_data)

    # Bindings must be device pointers
    bindings = [int(d_input), int(d_output)]
    
    context.execute_v2(bindings)
    cuda.memcpy_dtoh(output_data, d_output)
    return output_data


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

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


def decode_yolo_predictions(output, conf_threshold=0.5, iou_treshold=0.45, num_classes=494):
    """
    Decode YOLO predictions from (498, 435456) -> (435456, 498).
    
    Args:
        output: (498, 435456) or (435456, 498) array
        conf_threshold: Class score threshold
        intersection_over_union
        num_classes: Number of classes (494)
    
    Returns:
        boxes: (N, 4) in xyxy format
        confidences: (N,) - max class score
        class_ids: (N,)
    """
    # Reshape to (num_anchors, 498)
    if output.shape[0] == 498:
        output = output.T  # Now (435456, 498)
    
    # Extract components
    boxes_xywh = output[:, :4]  # (435456, 4)
    class_scores = output[:, 4:4+num_classes]  # (435456, 494)
    
    # Get class predictions
    class_ids = np.argmax(class_scores, axis=1)  # (435456,)
    confidences = np.max(class_scores, axis=1)  # (435456,) - max class score
    
    # Filter by confidence threshold
    valid = confidences >= conf_threshold
    
    boxes_xywh = boxes_xywh[valid]
    confidences = confidences[valid]
    class_ids = class_ids[valid]    
    
    # Convert from center format (x, y, w, h) to corner (x1, y1, x2, y2)
    x, y, w, h = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    
    boxes = np.stack([x1, y1, x2, y2], axis=1)
    
    if len(boxes) == 0:
        return boxes, confidences, class_ids
    
    # Sort by confidence descending
    sorted_idx = np.argsort(-confidences)
    boxes = boxes[sorted_idx]
    confidences = confidences[sorted_idx]
    class_ids = class_ids[sorted_idx]
    
    i = 0
    for i in range(len(boxes)):
        if i >= len(boxes) - 1:
            break
        
        # Compute IoU with remaining boxes
        iou = compute_iou(boxes[i], boxes[i+1:])  # (1, M)
        valid = np.logical_or(iou[0] <= iou_treshold, class_ids[i+1:] != class_ids[i])  # Keep if IoU is low or class is different 
        valid = [True] * (i + 1) + list(valid)
        boxes = boxes[valid]
        confidences = confidences[valid]
        class_ids = class_ids[valid]

    return boxes, confidences, class_ids


def compute_iou(box1, boxes2):
    """
    Compute IoU between two sets of boxes in xyxy format.
    
    Args:
        boxes1: (4) array in xyxy format
        boxes2: (M, 4) array in xyxy format
    
    Returns:
        iou: (N, M) IoU matrix
    """
    x1_min, y1_min, x1_max, y1_max = box1[0], box1[1], box1[2], box1[3]
    x2_min, y2_min, x2_max, y2_max = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]
    
    # Intersection
    inter_xmin = np.maximum(x1_min[np.newaxis, np.newaxis], x2_min[np.newaxis, :])
    inter_ymin = np.maximum(y1_min[np.newaxis, np.newaxis], y2_min[np.newaxis, :])
    inter_xmax = np.minimum(x1_max[np.newaxis, np.newaxis], x2_max[np.newaxis, :])
    inter_ymax = np.minimum(y1_max[np.newaxis, np.newaxis], y2_max[np.newaxis, :])
    
    inter_w = np.maximum(0, inter_xmax - inter_xmin)
    inter_h = np.maximum(0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h
    
    # Union
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1[np.newaxis, np.newaxis] + area2[np.newaxis, :] - inter_area
    
    iou = inter_area / (union_area + 1e-6)
    return iou


if __name__ == "__main__":
    # Paths
    model_version = "V2"
    region_id = "17T"
    pt_model_path = f"models/{model_version}/trained-ld/{region_id}/{region_id}_weights.pt"
    trt_engine_path = f"models/{model_version}/trained-ld/{region_id}/{region_id}_weights.trt"
    bbox_path = f"models/{model_version}/trained-ld/{region_id}/bounding_boxes.csv"
    image_path = f"models/{model_version}/sample_images/l8_{region_id}_00277.png"
    
    
    img = Image.open(image_path).convert("RGB")
    
    # Crop image to 4608x2592 and letterbox to 4608x4608
    img_array = np.array(img)
    height, width = img_array.shape[:2]

    # Crop to 4608x2592
    crop_width, crop_height = 4608, 2592
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    img_cropped = img_array[top:top + crop_height, left:left + crop_width]

    # Letterbox to 4608x4608
    target_size = 4608
    pad_top = (target_size - crop_height) // 2
    pad_bottom = target_size - crop_height - pad_top
    img_letterboxed = np.pad(img_cropped, ((pad_top, pad_bottom), (0, 0), (0, 0)), mode='constant', constant_values=0)

    img = Image.fromarray(img_letterboxed)
    
    batch_size = 1
    img_letterboxed = np.expand_dims(img_letterboxed, axis=0)
    # NCHW
    img_letterboxed = np.transpose(img_letterboxed, (0, 3, 1, 2)) / 255.0
    model = TrtModel(trt_engine_path)
    # shape = model.engine.get_tensor_shape(0)

    
    
    result = model(img_letterboxed,batch_size)
    result_array = result[0].squeeze()
    print(f"TensorRT output shape: {result_array.shape}")
    # nms_result_array = torchvision.ops.nms(torch.from_numpy(result_array), torch.ones(result_array.shape[0]), iou_threshold=0.5).numpy()
    trt_boxes, trt_confidences, trt_class_ids = decode_yolo_predictions(result_array, conf_threshold=0.5, num_classes=494)
    
    # engine = load_tensorrt_model(trt_engine_path)
    # trt_output = run_tensorrt_inference(engine, img_letterboxed)
    # max_val = np.max(trt_output)
    # print(f"Max value in TensorRT output: {max_val}")
    
    IMAGE_WIDTH = 2592
    IMAGE_HEIGHT = 4608
    model = YOLO(pt_model_path)
    ground_truth = np.loadtxt(bbox_path, delimiter=",", skiprows=1)
    
    # transformations = torchvision.transforms.Compose(
    #     [
    #         torchvision.transforms.Resize((1216, 1216)),
    #         torchvision.transforms.ToTensor(),
    #         # torchvision.transforms.Normalize(
    #         #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    #         # ),
    #     ]
    # )
    
    device = "cpu" # torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    
    # img = transformations(img).unsqueeze(0).to(device)
    # img = img.to(device)  # Add batch dimension and move to device

    print(f"Input tensor shape: {img.size}")
    
    results = model.predict(
        img,
        conf=0.5,
        imgsz=4608,
        verbose=True,
    )
    
    result = results[0]
    print(result)
    
    landmarks = result.boxes

    xywh = landmarks.xywh.cpu().numpy()
    class_ids = landmarks.cls.cpu().numpy().astype(int)
    confidences = landmarks.conf.cpu().numpy()

    valid_indices = (
        np.all(xywh >= 0, axis=1)
        & (xywh[:, 0] <= IMAGE_WIDTH - 1)
        & (xywh[:, 1] <= IMAGE_HEIGHT - 1)
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

    landmark_detections = LandmarkDetections(
        pixel_coordinates=xywh[:, :2],
        latlons=ground_truth[class_ids, :2],
        class_ids=class_ids,
        region_ids=np.array([region_id] * len(class_ids)),
        confidences=confidences,
    )
    
    # compare pth and trt results
    # Sort both by class_id
    pt_sort_idx = np.argsort(landmark_detections.class_ids)
    trt_sort_idx = np.argsort(trt_class_ids)
    
    pt_sorted = landmark_detections[pt_sort_idx]
    trt_sorted_class_ids = trt_class_ids[trt_sort_idx]
    trt_sorted_confidences = trt_confidences[trt_sort_idx]
    trt_sorted_boxes = trt_boxes[trt_sort_idx]
    
    max_len = max(len(pt_sorted), len(trt_sorted_class_ids))
    
    print("PyTorch vs TensorRT detections (sorted by class_id):")
    for i in range(max_len):
        if i < len(pt_sorted):
            print(f"PyTorch [{i}]: Class ID: {pt_sorted.class_ids[i]}, Confidence: {pt_sorted.confidences[i]:.4f}, Pixel Coords: {pt_sorted.pixel_coordinates[i]}, LatLon: {pt_sorted.latlons[i]}")
        if i < len(trt_sorted_class_ids):
            print(f"TensorRT [{i}]: Class ID: {trt_sorted_class_ids[i]}, Confidence: {trt_sorted_confidences[i]:.4f}, Box: {trt_sorted_boxes[i]}")
        print()
