#!/usr/bin/env bash

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

check_cudnn() {
    [ -f "/usr/include/cudnn.h" ] || ldconfig -p | grep -q libcudnn
}

setup_cudnn() {
    log_info "Setting up..."

    # for lambda, cudnn is installed in a non-standard location
    if [ ! -d "/usr/lib/python3/dist-packages/tensorflow/include/third_party/gpus/cudnn/include" ]; then
        for cudnn_so in /usr/lib/python3/dist-packages/tensorflow/libcudnn*; do
            [ -f "$cudnn_so" ] && sudo ln -sf "$cudnn_so" /usr/lib/x86_64-linux-gnu/
        done

        for cudnn_header in /usr/lib/python3/dist-packages/tensorflow/include/third_party/gpus/cudnn/include/*; do
            [ -f "$cudnn_header" ] && sudo ln -sf "$cudnn_header" /usr/include/
        done

        log_info "cuDNN configured"
        return 0
    else
        log_info "To install cuDNN and NCCL, run the following commands, then `uv sync`:"
        echo "wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb"
        echo "sudo dpkg -i cuda-keyring_1.1-1_all.deb"
        echo "sudo apt-get update"
        echo "sudo apt install libnccl2 libnccl-dev"
        echo "sudo apt-get install -y cudnn9-cuda-12"
        return 1
    fi
}

main() {
    log_info "Starting dory installation..."

    if ! command -v git &> /dev/null; then
        log_info "Installing git..."
        if command -v apt-get &> /dev/null; then
            sudo apt-get update && sudo apt-get install -y git
        else
            log_warn "Please install git manually"
            exit 1
        fi
    fi

    log_info "Cloning dory repository..."
    git clone https://github.com/keatonelvins/dory.git
    cd dory

    log_info "Installing/updating uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    echo "uv >=0.9.0 required to build, current: $(uv --version)"

    # Source uv environment
    [ -f "$HOME/.local/bin/env" ] && source $HOME/.local/bin/env

    check_cudnn || setup_cudnn

    log_info "Installing Python dependencies..."
    uv sync

    log_info "Installation complete!"
}

main
