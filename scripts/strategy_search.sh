#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: scripts/strategy_search.sh <query> [dimension]" >&2
  exit 2
fi

query="$1"
dimension="${2:-}"
if [[ -n "$dimension" ]]; then
  conda run -n HyperVault python -m framework.cli strategy-search \
    --query "$query" \
    --dimension "$dimension"
else
  conda run -n HyperVault python -m framework.cli strategy-search --query "$query"
fi
