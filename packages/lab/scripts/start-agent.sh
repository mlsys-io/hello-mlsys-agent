#!/usr/bin/env bash
# Start the student's Claude Code in an mlab workspace (the same as running ./claude there).
#   scripts/start-agent.sh WORKSPACE [claude arguments...]
set -euo pipefail
workspace="${1:?usage: start-agent.sh WORKSPACE [claude arguments...]}"; shift
exec "$workspace/claude" "$@"
