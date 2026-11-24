# Shared helpers for the mirror scripts. Source, don't execute.
set -euo pipefail

ROOT=$(git rev-parse --show-toplevel)
TARGETS="$ROOT/mirror/targets.tsv"

# Print the targets.tsv row for a prefix, or fail.
target_row() {
  local prefix=$1 row
  row=$(awk -F'\t' -v p="$prefix" '!/^#/ && $1 == p' "$TARGETS")
  [ -n "$row" ] || { echo "no mirror target for prefix '$prefix' in $TARGETS" >&2; return 1; }
  printf '%s\n' "$row"
}

# Every configured prefix, one per line.
all_prefixes() { awk -F'\t' '!/^#/ && NF >= 2 {print $1}' "$TARGETS"; }

field() { printf '%s\n' "$1" | cut -f"$2"; }

# The mirror-side history of <prefix> at <rev> (default HEAD). Deterministic:
# the same consolidated history always yields the same commit IDs.
split_sha() { git subtree split --prefix="$1" "${2:-HEAD}" 2>/dev/null; }

# Fetch over HTTPS (mirrors are public); push over SSH with a deploy key.
# MIRROR_URL overrides both for a single prefix (tests use a local bare repo).
fetch_url() { printf '%s\n' "${MIRROR_URL:-https://github.com/$1.git}"; }
push_url()  { printf '%s\n' "${MIRROR_URL:-git@github.com:$1.git}"; }
