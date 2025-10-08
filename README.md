Some scaffolding to easily install the fast version of nemo-automodel

Includes:
- ported megatron backend
- grouped gemm
- deep ep
- transformer engine
- flash attention

All from a single:
```bash
uv sync
```
Sanity check install with `uv run python -c "import deep_ep; print('Success')"`

### Install Prereqs

DeepEP makes install a little tricky. See the [docs](https://github.com/deepseek-ai/DeepEP/blob/main/third-party/README.md) for more details.

You will need the nvidia usual suspects! Verify this all run okay:
```bash
# basic cuda
nvidia-smi
# nvcc
nvcc --version 
# nvlink
nvidia-smi nvlink --status
# cudnn
ldconfig -p | grep cudnn
# nccl
ldconfig -p | grep nccl
# rdma headers
ldconfig -p | grep ibverbs
# make sure cuda is in LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH | tr ':' '\n' | grep cuda
```

If not, install before retrying `uv sync`. Here's what I need to do for my gcp env:
```bash
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y libibverbs-dev
sudo apt-get install -y cudnn9-cuda-12
sudo apt install libnccl2 libnccl-dev
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

#### Multinode

Set NVSHMEM_DIR dir
```bash
export NVSHMEM_DIR=$(uv run python -c "import importlib.util; print(importlib.util.find_spec('nvidia.nvshmem').submodule_search_locations[0])")
export LD_LIBRARY_PATH="${NVSHMEM_DIR}/lib:$LD_LIBRARY_PATH"
```

Enable NVSHMEM IBGDA support for infiniband:
```bash
echo 'options nvidia NVreg_EnableStreamMemOPs=1 NVreg_RegistryDwords="PeerMappingOverride=1;"' | sudo tee /etc/modprobe.d/nvidia.conf
sudo update-initramfs -u
sudo reboot
```

Verify IBGDA is enabled:
```bash
cat /proc/driver/nvidia/params | grep EnableStreamMemOPs
# Should show: EnableStreamMemOPs: 1
```