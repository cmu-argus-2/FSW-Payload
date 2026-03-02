sudo apt update
sudo apt upgrade -y
sudo apt install python3-libnvinfer*
sudo apt autoremove -y

python3 -m venv .venv --system-site-packages
source .venv/bin/

python3 -m pip install -U pip setuptools wheel packaging


TORCH_OK=$(python3 -c "import pkg_resources,sys; \
    sys.exit(0 if any(d.project_name=='torch' and d.version=='2.3.0' for d in pkg_resources.working_set) else 1)" \
    && echo true || echo false)

if ! TORCH_OK; then
    wget -O torch-2.3.0-cp310-cp310-linux_aarch64.whl https://nvidia.box.com/shared/static/mp164asf3sceb570wvjsrezk1p4ftj8t.whl
fi


TORCHVISION_OK=$(python3 -c "import pkg_resources,sys; \
    sys.exit(0 if any(d.project_name=='torchvision' and d.version=='0.18.0a0+6043bc2' for d in pkg_resources.working_set) else 1)" \
    && echo true || echo false)

if ! TORCHVISION_OK; then
    wget -O torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl https://nvidia.box.com/shared/static/xpr06qe6ql3l6rj22cu3c45tz1wzi36p.whl
fi

ONNXRT_OK=$(python3 -c "import pkg_resources,sys; \
    sys.exit(0 if any(d.project_name=='onnxruntime-gpu' and d.version=='1.19.0' for d in pkg_resources.working_set) else 1)" \
    && echo true || echo false)

if ! ONNXRT_OK; then
    wget -O onnxruntime_gpu-1.19.0-cp310-cp310-linux_aarch64.whl https://nvidia.box.com/shared/static/6l0u97rj80ifwkk8rqbzj1try89fk26z.whl
fi


pip install -r ./models/requirements.txt

unset PYTHONPATH
export PYTHONPATH="$VIRTUAL_ENV/lib/python$(python3 -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")')/site-packages:$PYTHONPATH"
export PYTHONPATH="$PYTHONPATH:/usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu/nvidia"

rm -f torch-2.3.0-cp310-cp310-linux_aarch64.whl torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl
rm -f onnxruntime_gpu-1.19.0-cp310-cp310-linux_aarch64.whl