#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  printf 'usage: %s <folder> <zip-file-name>\n' "$0" >&2
  exit 2
fi

folder=$1
zip_file=$2
output_path="$folder/$zip_file"

if [[ ! -d "$folder" ]]; then
  printf 'folder does not exist: %s\n' "$folder" >&2
  exit 1
fi

if [[ -e "$output_path" ]]; then
  printf 'refusing to overwrite existing file: %s\n' "$output_path" >&2
  exit 1
fi

mapfile -d '' parts < <(find "$folder" -name "$zip_file.part*" -print0 | sort -z -V)

if [[ ${#parts[@]} -eq 0 ]]; then
  printf 'no parts found for %s in %s\n' "$zip_file" "$folder" >&2
  exit 1
fi

cat "${parts[@]}" > "$output_path"
printf 'merged %s parts into %s\n' "${#parts[@]}" "$output_path"
