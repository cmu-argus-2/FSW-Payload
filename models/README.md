# NN-models

This repositiory contains the trained region classification and landmark detection networks.

## Installation

Installation can be done by running the command:

```bash
sh python.sh
```

## Allocating swap memory on the NVMe

```bash
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

To make swap persistent across reboots:
```bash
sudo nano /etc/fstab
```

Add this line:

```
/swapfile none swap sw 0 0
```

Confirm:

```bash
free -h
```

## Jetson Stats

Jtop can be a useful tool to track memory/cpu/gpu usage when running the inference models. Link: https://github.com/rbonghi/jetson_stats

```bash
sudo apt update
sudo apt install python3-pip python3-setuptools -y
sudo pip3 install -U jetson-stats
```

