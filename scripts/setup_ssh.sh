#!/bin/bash
# Setup SSH key for Vast.ai GPU instances

KEY_DIR="$(dirname "$0")/../keys"
KEY_PATH="$KEY_DIR/vast_gpu"

mkdir -p "$KEY_DIR"

if [ -f "$KEY_PATH" ]; then
    echo "SSH key already exists: $KEY_PATH"
else
    echo "Generating new SSH key..."
    ssh-keygen -t ed25519 -f "$KEY_PATH" -N "" -C "vast-gpu"
    echo "SSH key created: $KEY_PATH"
fi

echo ""
echo "Public key (add to Vast.ai):"
echo "----------------------------------------"
cat "$KEY_PATH.pub"
echo "----------------------------------------"
echo ""
echo "To add this key to Vast.ai, run:"
echo "  cd $(dirname "$0")/.."
echo "  ./venv/bin/python -c \\"
echo "    from vast_gpu_manager import VastGPUManager; \\"
echo "    manager = VastGPUManager(); \\"
echo "    key = open('keys/vast_gpu.pub').read(); \\"
echo "    manager.add_ssh_key(key)"
echo ""
echo "To connect to an instance:"
echo "  ssh -i $KEY_PATH -p <port> root@<host>"
