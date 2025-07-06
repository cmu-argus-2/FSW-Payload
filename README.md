# Payload Flight Software for Argus

This repository contains the Argus Payload Flight Software for the Jetson Orin Nano (8GB) and its custom carrier board. Argus is a technology demonstration mission focused on vision-based orbit determination.

## Install instructions 

- CUDA
- TensorRT
- [spdlog](https://github.com/gabime/spdlog) - Logging library
- [OpenCV](https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html?ref=wasyresearch.com) - Computer vision library
- [Eigen3](http://eigen.tuxfamily.org/index.php?title=Main_Page#Download) - Linear algebra Library
- [GTest](https://github.com/google/googletest) - testing framework
- TODO

```bash
sudo chmod +x install_deps.sh build.sh run.sh 
./install_deps.sh
```

## Build instructions

Give permissions to both scripts 
```bash
sudo chmod +x build.sh run.sh 
```

### Setting up the Environment

To compile using CMake, build the project using the helper script:

```bash
./build.sh [Debug|Release] [disable-nn]
./bin/PAYLOAD [optional: <communication-interface: [UART, CLI]>] [optional: <log-severity: [TRACE, DEBUG, INFO, WARN, ERROR, CRITICAL, OFF]>]
```

All binaries will appear in the bin folder.

## Configuration

The configuration file is located at config/config.toml. Update this file to modify system parameters.

## Local interaction with the FSW

As a functional debugging tool, the Payload can be run and controlled locally through a command-line interface, either through the compiled one or the python interface (perfect replica). First, run the main process in CLI mode:

```bash
./bin/PAYLOAD [optional: <communication-interface: [UART, CLI]>] // default to UART 
```
In another terminal, launch either the compiled CLI or its python equivalent (perfect replica):
```bash
./bin/CLI_CMD
or
python python/cli_cmd.py
```

Under the hood, both processes communicates through 2 distinct named pipes (FIFO).

## Command-based paradigm 

The Payload communicates through its host machine via UART (transition from SPI in progress) with a set of predefined commands available at TODO (internal README).

