#!/usr/bin/env bash
# One-time setup for hello-mlsys-agent: Python workspace (simulator + lab) and the pinned Claude Code.
exec "$(dirname "${BASH_SOURCE[0]}")/../packages/lab/scripts/setup.sh" "$@"
