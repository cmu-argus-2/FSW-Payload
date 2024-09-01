# Payload Flight Software for Argus-1

In progress

## Requirements

- [spdlog](https://github.com/gabime/spdlog)
- [OpenCV](https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html?ref=wasyresearch.com)
- [Eigen3](http://eigen.tuxfamily.org/index.php?title=Main_Page#Download)
- [Torch](https://pytorch.org/get-started/locally/)
- [GTest](https://github.com/google/googletest)

## Build instructions

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
./build/PAYLOAD
```

or 

```bash
mkdir build
cd build
cmake .. && make
./PAYLOAD
```
