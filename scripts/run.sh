#!/usr/bin/env bash
# Launch N local processes with torchrun. macOS-safe: binds rendezvous + gloo to loopback
# (plain `torchrun --standalone` hangs on macOS resolving the IPv6 hostname).
# Usage: scripts/run.sh <nproc> <script.py> [args...]
set -euo pipefail
NPROC=${1:?usage: scripts/run.sh <nproc> <script.py> [args...]}; shift
cd "$(dirname "$0")/.."
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-lo0}
exec .venv/bin/torchrun --nnodes=1 --nproc_per_node="$NPROC" \
  --master_addr=127.0.0.1 --master_port="${MASTER_PORT:-29500}" "$@"
