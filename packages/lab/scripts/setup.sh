#!/usr/bin/env bash
# Create the Python environment and install the pinned Claude Code into it.
# Works in the hello-mlsys-agent workspace (one .venv at the repository root, simulator included)
# and in a standalone hello-mlsys-lab checkout.
set -euo pipefail
lab="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$lab"
command -v uv >/dev/null || { echo "uv not found. Install it first: https://docs.astral.sh/uv/" >&2; exit 1; }
if [ -f ../../pyproject.toml ] && grep -q '^\[tool.uv.workspace\]' ../../pyproject.toml; then
    (cd ../.. && uv sync --all-packages --extra dev --extra web --extra agent --extra simulator --python 3.12)
else
    # The simulator extra is only available in the workspace. Install the requested extras alone.
    uv run --no-sync --python 3.12 uv pip install --editable '.[dev,web,agent]'
fi
python="$(uv run --no-sync python -c 'import sys; print(sys.executable)')"
prefix="$("$python" -c 'import sys; print(sys.prefix)')"
version="$("$python" -c 'from mlab.core.io import pinned_claude_version; print(pinned_claude_version())')"
installed="$("$prefix/bin/claude" --version 2>/dev/null | awk '{print $1}' || true)"
if [ "$installed" != "$version" ]; then
    command -v npm >/dev/null || { echo "Install Node.js/npm to install Claude Code $version into this project." >&2; exit 1; }
    npm install --silent --no-fund --no-audit --cache "$prefix/cache/npm" --prefix "$prefix/lib/mlab-claude" \
        "@anthropic-ai/claude-code@$version"
    ln -sfn ../lib/mlab-claude/node_modules/.bin/claude "$prefix/bin/claude"
fi
echo "Ready: Claude Code $("$prefix/bin/claude" --version | awk '{print $1}') in $prefix/bin."
echo "Student agent: export QWEN38_BASE_URL (OpenAI-compatible, ends in /v1), QWEN38_MODEL and, if needed, QWEN38_API_KEY;"
echo "then: mlab doctor --live && mlab init my-lab --task 9.2 && cd my-lab && ./claude"
