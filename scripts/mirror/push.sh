#!/usr/bin/env bash
# Push <prefix>'s split history (and its release tags) to the read-only mirror.
#   scripts/mirror/push.sh <prefix> [--dry-run]
# Never forces. A rejected push means someone committed to the mirror directly:
# the job fails, and scripts/mirror/absorb-upstream.sh brings their commits in.
source "$(dirname "$0")/lib.sh"
cd "$ROOT"
prefix=${1:?usage: push.sh <prefix> [--dry-run] [--tags-only]}; shift
dry=; tags_only=
for flag in "$@"; do case "$flag" in --dry-run) dry=--dry-run ;; --tags-only) tags_only=1 ;; esac; done
row=$(target_row "$prefix")
remote=$(push_url "$(field "$row" 2)"); branch=$(field "$row" 3); tag_prefix=$(field "$row" 4)
sha=$(split_sha "$prefix")
[ -n "$sha" ] || { echo "empty split for $prefix" >&2; exit 1; }
# On a tag event HEAD is the tagged commit, which may be behind main: push only the tags then.
refspecs=()
[ -n "$tags_only" ] || refspecs+=("$sha:refs/heads/$branch")
# Tags: sim-v1.25.0 on the consolidated repo -> v1.25.0 on the mirror, pointing
# at the split of the tagged commit.
while read -r tag; do
  [ -n "$tag" ] || continue
  refspecs+=("$(split_sha "$prefix" "$tag^{commit}"):refs/tags/v${tag#"$tag_prefix"}")
done < <(git tag --list "${tag_prefix}*")
echo "mirror $prefix -> $remote"
[ ${#refspecs[@]} -gt 0 ] || { echo "  nothing to push"; exit 0; }
printf '  %s\n' "${refspecs[@]}"
args=(--porcelain)
[ -n "$dry" ] && args+=(--dry-run)
# No --force anywhere: a non-fast-forward is rejected by git and fails this script.
git push "${args[@]}" "$remote" "${refspecs[@]}"
