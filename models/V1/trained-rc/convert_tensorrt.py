import sys
import tensorrt as trt

def build_engine(model_path):

    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) 
    builder = trt.Builder(TRT_LOGGER)

    ##### Network definition 
    # https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#explicit-implicit-batch
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)

    ##### Parse the ONNX model
    parser = trt.OnnxParser(network, TRT_LOGGER) 
    with open(model_path, "rb") as f:
        if not parser.parse(f.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            sys.exit(1)  # Proper exit
        else:
            print("ONNX parse ended successfully")

    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    print(f"Inputs: {inputs[0].name}")
    print(f"Outputs: {outputs[0].name}")
    #exit()

    ##### Build the engine
    config = builder.create_builder_config()
    max_workspace_size = 1 << 30  # 1 GiB (1024 MB)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size)
    config.default_device_type = trt.DeviceType.GPU
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED # for now, debuggign purposes
    # config.profiling_verbosity = trt.ProfilingVerbosity.LAYER_NAMES_ONLY
    # config.profiling_verbosity = trt.ProfilingVerbosity.NONE # for final sat
    
    ##### Create an optimization profile for dynamic shapes ~ should be fixed for our purposes
    profile = builder.create_optimization_profile()
    # network.add_input("input", trt.float32, (-1, 3, -1, -1))  # Fix channels to 3 (RGB), -1 indicates dynamic
    ##### input shape ranges definition
    profile.set_shape(
        "input",
        (1, 3, 224, 224),  # Min: Smallest possible input (standard EfficientNetB0 input)
        (1, 3, 224, 224),  # Opt: Model's default size (what the model optimize for)
        (8, 3, 512, 512)  # Max: Largest expected input
    )
    config.add_optimization_profile(profile)

    # config.set_flag(trt.BuilderFlag.FP16) # Enable FP16 mode - No need for now
    # config.set_flag(trt.BuilderFlag.INT8) # Enable INT8 mode - No need for now

    print(f"config = {config}")
    print("====================== Building TensorRT Engine... ======================")

    ##### Ensure engine is built and serialized safely
    with builder.build_serialized_network(network, config) as engine:
        if engine is None:
            print("Error: Failed to create TensorRT engine.")
            sys.exit(1)

        print("Engine created successfully")

        try:
            with open("model.trt", 'wb') as f:
                f.write(bytearray(engine))
        except Exception as e:
            print(f"Error writing the engine to file: {e}")
    
    return engine

if __name__ == '__main__':
    build_engine( model_path="effnet.onnx")