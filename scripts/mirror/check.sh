#!/usr/bin/env bash
# Dry run: would pushing <prefix> (default: all) to its mirror be a fast-forward?
#   scripts/mirror/check.sh [prefix ...]
# Exit 0: every mirror is equal to or behind the split. Exit 1: a mirror has
# commits the consolidated repo lacks (run absorb-upstream.sh), or is unreachable.
# Read-only: fetches into a scratch ref namespace, never pushes.
source "$(dirname "$0")/lib.sh"
cd "$ROOT"
status=0
if [ $# -gt 0 ]; then prefixes=("$@"); else mapfile -t prefixes < <(all_prefixes); fi
for prefix in "${prefixes[@]}"; do
  row=$(target_row "$prefix"); remote=$(fetch_url "$(field "$row" 2)"); branch=$(field "$row" 3)
  local_sha=$(split_sha "$prefix")
  ref="refs/mirror-check/${prefix//\//_}"
  if ! git fetch --quiet --no-tags "$remote" "+$branch:$ref" 2>/dev/null; then
    echo "UNREACHABLE $prefix -> $remote ($branch)"; status=1; continue
  fi
  remote_sha=$(git rev-parse "$ref")
  if [ "$remote_sha" = "$local_sha" ]; then
    echo "IN-SYNC     $prefix  $local_sha"
  elif git merge-base --is-ancestor "$remote_sha" "$local_sha"; then
    echo "FAST-FWD    $prefix  $remote_sha -> $local_sha ($(git rev-list --count "$remote_sha..$local_sha") commits)"
  else
    echo "DIVERGED    $prefix  mirror $remote_sha is not an ancestor of split $local_sha"
    echo "            run: scripts/mirror/absorb-upstream.sh $prefix"
    status=1
  fi
done
exit $status
