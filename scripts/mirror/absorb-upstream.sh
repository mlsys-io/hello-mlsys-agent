#!/usr/bin/env bash
# Bring commits made directly on a mirror back into the consolidated repo.
#   scripts/mirror/absorb-upstream.sh <prefix>
# Runs `git subtree pull` (full history, no --squash) on a new branch, so the
# next push.sh is a fast-forward again. Open a PR from the branch it prints.
source "$(dirname "$0")/lib.sh"
cd "$ROOT"
prefix=${1:?usage: absorb-upstream.sh <prefix>}
row=$(target_row "$prefix"); remote=$(fetch_url "$(field "$row" 2)"); branch=$(field "$row" 3)
[ -z "$(git status --porcelain)" ] || { echo "working tree not clean" >&2; exit 1; }
work="mirror/absorb-${prefix//\//-}-$(date -u +%Y%m%d%H%M%S)"
git switch -c "$work"
git subtree pull --prefix="$prefix" "$remote" "$branch" \
  -m "Absorb direct commits from the $prefix mirror ($remote $branch)"
echo "absorbed into branch $work; verify with scripts/mirror/check.sh $prefix, then open a PR"
