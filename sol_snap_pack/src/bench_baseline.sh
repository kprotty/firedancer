#!/bin/bash
# bench_baseline.sh <stream-dir> -- compress every stream with zstd at a few
# levels and print a table.  zstd -1 is what firedancer itself uses for
# snapshots (FD_BACKUP_ZSTD_LEVEL), so that column is the number to beat.
set -u
dir=$1
levels=${2:-"1 3 9 19"}
printf "%-16s %14s" "stream" "raw"
for l in $levels; do printf " %14s" "zstd-$l"; done
printf "\n"
tot_raw=0
declare -A tot
for l in $levels; do tot[$l]=0; done
for f in "$dir"/*.bin; do
  [ "$(basename "$f")" = whole.bin ] && continue
  b=$(basename "$f" .bin)
  raw=$(stat -c %s "$f")
  printf "%-16s %14d" "$b" "$raw"
  tot_raw=$((tot_raw+raw))
  for l in $levels; do
    c=$(zstd -q -$l -c --long=27 -T1 < "$f" | wc -c)
    printf " %14d" "$c"
    tot[$l]=$(( ${tot[$l]} + c ))
  done
  printf "\n"
done
printf "%-16s %14d" "TOTAL(fields)" "$tot_raw"
for l in $levels; do printf " %14d" "${tot[$l]}"; done
printf "\n"
# whole.bin reported separately: it is the array-of-structs baseline
if [ -f "$dir/whole.bin" ]; then
  raw=$(stat -c %s "$dir/whole.bin")
  printf "%-16s %14d" "AoS baseline" "$raw"
  for l in $levels; do printf " %14d" "$(zstd -q -$l -c --long=27 -T1 < "$dir/whole.bin" | wc -c)"; done
  printf "\n"
fi
