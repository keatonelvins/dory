#!/usr/bin/env bash

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }

check_cudnn() {
    [ -f "/usr/include/cudnn.h" ] || ldconfig -p | grep -q libcudnn
}

setup_cudnn_lambda() {
    log_info "Setting up cuDNN for Lambda environment..."

    if [ ! -d "/usr/lib/python3/dist-packages/tensorflow/include/third_party/gpus/cudnn/include" ]; then
        log_warn "cuDNN not found, transformer-engine build may fail"
        return 1
    fi

    for cudnn_so in /usr/lib/python3/dist-packages/tensorflow/libcudnn*; do
        [ -f "$cudnn_so" ] && sudo ln -sf "$cudnn_so" /usr/lib/x86_64-linux-gnu/
    done

    for cudnn_header in /usr/lib/python3/dist-packages/tensorflow/include/third_party/gpus/cudnn/include/*; do
        [ -f "$cudnn_header" ] && sudo ln -sf "$cudnn_header" /usr/include/
    done

    log_info "cuDNN configured"
}

main() {
    log_info "Starting dory installation..."

    check_cudnn || setup_cudnn_lambda

    log_info "Installing/updating uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    [ -f "$HOME/.local/bin/env" ] && source $HOME/.local/bin/env

    log_info "Installing dependencies (uv $(uv --version))..."
    uv sync

    log_info "Installation complete!"
}

main
