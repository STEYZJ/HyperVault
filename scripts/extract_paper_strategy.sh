#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: scripts/extract_paper_strategy.sh <paper-id-or-path>" >&2
  exit 2
fi

conda run -n HyperVault python -m framework.cli extract-paper-strategy --paper "$1"
