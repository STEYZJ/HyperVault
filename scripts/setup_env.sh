#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! conda env list | awk '{print $1}' | grep -qx "HyperVault"; then
  conda create -n HyperVault python=3.12 -y
fi

conda run -n HyperVault python -m pip install --upgrade pip
conda run -n HyperVault python -m pip install -r "${PROJECT_ROOT}/requirements.txt"

echo "HyperVault environment is ready."

