Playground for the fast version of nemo-automodel

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

### nvidia prereqs for DeepEP

DeepEP makes install a little tricky. See the [docs](https://github.com/deepseek-ai/DeepEP/blob/main/third-party/README.md) for more details.

You will need the nvidia usual suspects! Verify these all run okay:
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

If not, install before retrying `uv sync`. Assuming at least CUDA toolkit is installed, you may also need to:
```bash
# install nccl from source
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt install libnccl2 libnccl-dev

# install rmda headers and cudnn from meta-packages
sudo apt-get install -y libibverbs-dev
sudo apt-get install -y cudnn9-cuda-12

# make sure path to CUDA toolkit install is on LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
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