#!/usr/bin/env bash
set -euo pipefail

conda run -n HyperVault python -m framework.cli openai-smoke
