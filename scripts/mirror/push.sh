#!/usr/bin/env bash
# Push <prefix>'s split history (and its release tags) to the read-only mirror.
#   scripts/mirror/push.sh <prefix> [--dry-run]
# Never forces. A rejected push means someone committed to the mirror directly:
# the job fails, and scripts/mirror/absorb-upstream.sh brings their commits in.
source "$(dirname "$0")/lib.sh"
cd "$ROOT"
prefix=${1:?usage: push.sh <prefix> [--dry-run]}; dry=${2:-}
row=$(target_row "$prefix")
remote=$(push_url "$(field "$row" 2)"); branch=$(field "$row" 3); tag_prefix=$(field "$row" 4)
sha=$(split_sha "$prefix")
[ -n "$sha" ] || { echo "empty split for $prefix" >&2; exit 1; }
refspecs=("$sha:refs/heads/$branch")
# Tags: sim-v1.25.0 on the consolidated repo -> v1.25.0 on the mirror, pointing
# at the split of the tagged commit.
while read -r tag; do
  [ -n "$tag" ] || continue
  refspecs+=("$(split_sha "$prefix" "$tag^{commit}"):refs/tags/v${tag#"$tag_prefix"}")
done < <(git tag --list "${tag_prefix}*")
echo "mirror $prefix -> $remote"
printf '  %s\n' "${refspecs[@]}"
args=(--porcelain)
[ "$dry" = "--dry-run" ] && args+=(--dry-run)
# No --force anywhere: a non-fast-forward is rejected by git and fails this script.
git push "${args[@]}" "$remote" "${refspecs[@]}"
