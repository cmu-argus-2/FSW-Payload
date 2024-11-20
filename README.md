# Payload Flight Software for Argus-1

This repository contains the Flight Software (FSW) written for the Jetson Orin Nano (8GB) and its custom carrier board. Argus-1 is a technology demonstration mission focused on vision-based orbit determination.


## Requirements

- [spdlog](https://github.com/gabime/spdlog) - Logging library
- [OpenCV](https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html?ref=wasyresearch.com) - Computer vision library
- [Eigen3](http://eigen.tuxfamily.org/index.php?title=Main_Page#Download) - Linear algebra Library
- [Torch](https://pytorch.org/get-started/locally/) - Deep learning 
- [GTest](https://github.com/google/googletest) - testing framework

## Build instructions

### Setting up the Environment

To compile using CMake, follow these steps: 

Set the `LIBTORCH_PATH` environment variable in your `.bashrc`:
   ```bash
   export LIBTORCH_PATH=/path/to/libtorch/share/cmake/Torch/
   ```

In your terminal:

    ```bash
    source ~/.bashrc
    ```

Build the project

```bash
./build.sh
./build/PAYLOAD [optional: <communication-interface: [UART, CLI]>]
```

or 

```bash
mkdir build
cd build
cmake .. && make
./PAYLOAD [optional: <communication-interface: [UART, CLI]>] // default to UART 
```
## Configuration

The configuration file is located at config/config.toml. Update this file to modify system parameters.

## Local interaction with the FSW

As a functional debugging tool, the Payload can be run and controlled locally through a named pipe (FIFO) given to the Payload and the command line interface. For this control mode, the command line interface must be run first:

```bash
./CLI_CMD
./PAYLOAD [optional: <communication-interface: [UART, CLI]>] // default to UART 
```

## Command-based paradigm 

The Payload communicates through its host machine via UART (transition from SPI in progress) with a set of predefined commands available at TODO (internal README).


## General Architecture 

TODO