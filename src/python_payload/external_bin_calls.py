"""
This file will implement the code that will call the c++ bins to perform the actions
like record a dataset, or run inference...
"""

import os
import subprocess





def run_inference(img_path, output_folder_path):
    """
    will run the inference on a given image
    it will also give a path for the generated data to be saved
    for now this will be running on the payload folder that is on Documents
    will move to the run path and run the binary from there 
    """
    
    run_path = "/home/argus-payload/Documents/FSW-Payload"
    bin_name = "./bin/RUN_INFERENCE"
    
    rc_model = "models/V1/trained-rc/effnet_0997acc.trt"
    ld_model = "models/V1/trained-ld"
    
    print(f"Running inference on {img_path}")
    print(f"Output folder: {output_folder_path}")
    print(f"RC model: {rc_model}")
    print(f"LD model: {ld_model}")
    
    result = subprocess.run([bin_name, rc_model, ld_model, img_path, output_folder_path],
        cwd=run_path,
        # capture_output=True,
        text=True
    )
    
    # TODO check if it has completed successfully
    
    return True
    