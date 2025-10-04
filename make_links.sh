#!/usr/bin/env bash
# Usage: ./make_links.sh /path/to/src_dir /path/to/dst_dir <threshold>
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 <source_dir> <dest_dir> <dt> <last_day>" >&2
  exit 1
fi

src_dir=$1
dst_dir=$2
dt=$3
threshold=$4

mkdir -p "$dst_dir"

# Resolve src_dir to an absolute path so symlinks always point correctly
src_dir_abs=$(cd "$src_dir" && pwd)

shopt -s nullglob
for path in "$src_dir_abs"/*.data; do
  file=$(basename "$path")

  # Match: "<F>.<N>.data" where N is exactly 10 digits
  if [[ $file =~ ^(.*)\.([0-9]{10})\.data$ ]]; then
    F="${BASH_REMATCH[1]}"
    N="${BASH_REMATCH[2]}"

    # Convert N to decimal explicitly
    N_dec=$((10#$N))

    # Skip if N is greater than the threshold
    if (( N_dec > (threshold*24*60*60/dt) )); then
      echo "skipping (N=$N_dec beyond day $threshold): $file"
      continue
    fi

    # Compute M = N * dt / 60 / 10 - M is multiplier of 10 minutes.
    M=$((N_dec * dt / 600))

    # If you want exactly 10 digits for M, uncomment this:
    printf -v M "%010d" "$M"

    link_path="$dst_dir/$F.$M.data"

    # Create or replace the symlink
    ln -sfn "$path" "$link_path"
    echo "linked: $link_path -> $path"
  fi
done
