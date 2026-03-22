import onnx
import onnx_graphsurgeon as gs
import numpy as np
from typing import *
import os
from pathlib import Path

# Adapted from https://medium.com/@MaroJEON/batched-nms-1-yolov8-model-modification-without-modeler-using-onnx-graphsurgeon-d876b75478af
class OnnxModifier:
    def __init__(self, onnx_path, fp16=False):
        self.graph = gs.import_onnx(onnx.load(onnx_path))
        self.tensor = self.graph.tensors()
        self.nodes = self.graph.nodes
        self.fp16 = fp16
        
    def remove_nodes(self, remove_node_list: List[str]):
        for remove_nd in remove_node_list:
            removed = [node for node in self.nodes if node.name == remove_nd][0]
            self.nodes.remove(removed)
        
        output_tensor = self.graph.outputs[0]
        
        self.graph.outputs.remove(output_tensor)
    
    def add_transpose_nodes(self, input_tensor, perm: List[int], fp16=False):
        attrs = {"perm": perm}
        transpose_inputs = [input_tensor]
        if input_tensor.shape is not None:
            output_shape = [input_tensor.shape[i] for i in perm]
            # output_shape [*input_tensor.shape[:-1]]
        else:
            output_shape = None
        
        if fp16:
            dtype = np.float16
        else:            
            dtype = np.float32
            
        transpose_outputs = [gs.Variable(name="%s_transpose_output"%(input_tensor.name), dtype=dtype, shape=output_shape)]
            
        transpose_node = gs.Node(op="Transpose", name="%s_transpose"%(input_tensor.name),inputs=transpose_inputs, outputs=transpose_outputs, attrs=attrs)
        
        self.nodes.append(transpose_node)
        
        return transpose_node.outputs[0]
    
    def add_slice_node(self, sig_output_tensor, starts=5, ends=np.iinfo(np.int64).max, axes=4, step=1):
        data_input = sig_output_tensor
        
        starts_input = gs.Constant(name="%s_%d_%d_starts_Constant"%(data_input.name,starts,axes), values = np.array([starts])) # 5 is starts point
        ends_input = gs.Constant(name="%s_%d_%d_ends_Constant"%(data_input.name,starts,axes), values = np.array([ends]))
        axes_input = gs.Constant(name="%s_%d_%d_axes_Constant"%(data_input.name,starts,axes), values = np.array([axes]))
        step_input = gs.Constant(name="%s_%d_%d_steps_Constant"%(data_input.name,starts,axes), values = np.array([step]))

        slice_inputs = [data_input, starts_input, ends_input, axes_input, step_input]
        slice_shape = sig_output_tensor.shape.copy()
        if slice_shape is not None:
            if ends == np.iinfo(np.int64).max:
                ends = sig_output_tensor.shape[2]
            if isinstance(ends,str): # dynamic input onnx network - used to export rectangular TRT
                slice_shape[axes] = ends
            else:
                slice_shape[axes] = (ends - starts + step - 1) // step

        slice_outputs = [gs.Variable(name="%s_%d_%dslice_output_0"%(data_input.name, starts, axes), dtype=data_input.dtype, shape=slice_shape)]
        
        slice_node = gs.Node(op="Slice", name="%s_%d_%d_slice"%(data_input.name, starts, axes), inputs=slice_inputs, outputs=slice_outputs)
        
        self.nodes.append(slice_node)   
         
        return slice_node.outputs[0]
    
    def carve_output(self, output_list):
        self.graph.outputs = output_list
    
    def add_yolo_output(self, tensor, output_name):
        input_node = tensor.inputs[0]
        if output_name == "bbox":
            input_node.outputs = [gs.Variable(name=output_name).to_variable(dtype=input_node.outputs[0].dtype, shape = ["batch", None, 4])]
            
        elif output_name == "conf":
            input_node.outputs = [gs.Variable(name=output_name).to_variable(dtype=input_node.outputs[0].dtype, shape = ["batch", None])]
            
        elif output_name == "class_id":
            input_node.outputs = [gs.Variable(name=output_name).to_variable(dtype=input_node.outputs[0].dtype, shape = ["batch", None])]
            
        return input_node.outputs[0]
    
    def add_nms_plugin_nodes(self, bbox_output, score_output, fp16=False, class_agnostic=False, keep_topk=500, score_threshold=0.5, iou_threshold=0.45):
        # Documentation at: https://github.com/NVIDIA/TensorRT/blob/main/plugin/efficientNMSPlugin/README.md
        # float	score_threshold *	The scalar threshold for score (low scoring boxes are removed).
        # float	iou_threshold	The scalar threshold for IOU (additional boxes that have high IOU overlap with previously selected boxes are removed).
        # int	max_output_boxes	The maximum number of detections to output per image.
        # int	background_class	The label ID for the background class. If there is no background class, set it to -1.
        # bool	score_activation *	Set to true to apply sigmoid activation to the confidence scores during NMS operation.
        # bool	class_agnostic	Set to true to do class-independent NMS; otherwise, boxes of different classes would be considered separately during NMS.
        # int	box_coding	Coding type used for boxes (and anchors if applicable), 0 = BoxCorner (xyxy), 1 = BoxCenterSize (xywh).
        
        attrs = {}
        attrs["class_agnostic"] = int(class_agnostic)
        attrs["background_class"] = -1
        attrs["score_activation"] = 0
        attrs["max_output_boxes"] = keep_topk
        attrs["score_threshold"] = score_threshold
        attrs["iou_threshold"] = iou_threshold
        attrs["box_coding"] = 1 # what is this?
        attrs["plugin_version"] = "1" 

        batch_size = self.graph.inputs[0].shape[0]
        input_h = self.graph.inputs[0].shape[2]
        input_w = self.graph.inputs[0].shape[3]
        
        if fp16:
            dtype = np.float16
        else:
            dtype = np.float32
            
        num_detections = gs.Variable(name="num_detections").to_variable(dtype=np.int32, shape=[batch_size, 1])
        nmsed_boxes = gs.Variable(name="bbox").to_variable(dtype=dtype, shape=[batch_size, keep_topk, 4])
        nmsed_scores = gs.Variable(name="conf").to_variable(dtype=dtype, shape=[batch_size, keep_topk])
        nmsed_classes = gs.Variable(name="class_id").to_variable(dtype=np.int32, shape=[batch_size, keep_topk])

        nms_outputs = [num_detections, nmsed_boxes, nmsed_scores, nmsed_classes]

        nms_node = gs.Node(
            op="EfficientNMS_TRT",
            attrs=attrs,
            inputs=[bbox_output, score_output],
            outputs=nms_outputs)

        self.nodes.append(nms_node)

        return nms_node.outputs


if __name__ == "__main__":
    trained_ld_path = "models/V1/trained-ld"
    fp16 = True
    class_agnostic = False # default yolo option
    keep_topk = 300 # default yolo value
    score_threshold = 0.5
    iou_threshold = 0.45
    imgsz = 4608
    for folder in os.listdir(trained_ld_path):
        folder_path = os.path.join(trained_ld_path, folder)
        
        if not os.path.isdir(folder_path):
            continue
        
        # Find ONNX files in the folder
        onnx_load_path = os.path.join(folder_path, folder + "_weights.onnx")
        
        if fp16:
            onnx_load_path = onnx_load_path.replace(".onnx", "_fp16.onnx")
        else:
            onnx_load_path = onnx_load_path.replace(".onnx", "_fp32.onnx")
        
        if imgsz is not None:
            onnx_load_path = onnx_load_path.replace(".onnx", f"_sz_{imgsz}.onnx")
        
        if not os.path.exists(onnx_load_path):
            print(f"Skipping {folder}: no ONNX model found")
            continue
        
        print(f"Processing {folder}: {onnx_load_path}")
        
        try:
            onnx_md = OnnxModifier(onnx_load_path)
            output_tensor = onnx_md.graph.outputs[0]

            transposed_output = onnx_md.add_transpose_nodes(output_tensor, [0, 2, 1], fp16=fp16)
            # ---- bbox ---- #
            bbox_out_tensor = onnx_md.add_slice_node(transposed_output, starts=0, ends=4, axes=2, step=1)
            # ---- score ---- #
            conf_score_out_tensor = onnx_md.add_slice_node(transposed_output, starts=4, axes=2, step=1)

            # ---- make last node ---- #
            bbox_last_output = onnx_md.add_yolo_output(bbox_out_tensor, "nms_bbox")
            score_last_output = onnx_md.add_yolo_output(conf_score_out_tensor, "nms_score")

            # ---- import batchedNMS_TRT plugin ---- #
            num_detections, nmsed_boxes, nmsed_scores, nmsed_classes = \
                            onnx_md.add_nms_plugin_nodes(bbox_last_output, score_last_output, 
                                                        fp16=fp16, class_agnostic=class_agnostic, keep_topk=keep_topk, 
                                                        score_threshold=score_threshold, iou_threshold=iou_threshold)

            # ---- make nms plugin outputs ---- #
            onnx_md.carve_output([num_detections, nmsed_boxes, nmsed_scores, nmsed_classes])

            onnx_md.graph.cleanup().toposort()
            output_path = onnx_load_path = onnx_load_path.replace(".onnx", "_nms.onnx")
            onnx.save(gs.export_onnx(onnx_md.graph), output_path)
            print(f"Saved to {output_path}")
        except Exception as e:
            print(f"Error processing {folder}: {e}")
