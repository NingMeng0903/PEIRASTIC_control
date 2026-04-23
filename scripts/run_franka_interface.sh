#!/bin/bash
# 启动 franka-interface，自动设置库路径（lib/ 和根目录均有 .so）
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export LD_LIBRARY_PATH="${ROOT}/lib:${ROOT}:${LD_LIBRARY_PATH}"

CONFIG_PATH="${1:-${ROOT}/config/local-host.yml}"
if [ $# -gt 0 ]; then
  shift
fi

if [ -x "${ROOT}/build/franka-interface" ]; then
  exec "${ROOT}/build/franka-interface" "${CONFIG_PATH}" "$@"
fi

exec "${ROOT}/bin/franka-interface" "${CONFIG_PATH}" "$@"
