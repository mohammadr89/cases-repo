#!/bin/bash

# Source directory containing the files to copy
source_dir=~/maarten_nh3_main
# source_dir=~/case_base

# Find all the innermost directories
find . -type d -path "*/*/*/bi_dir" -o -path "*/*/*/one_dir" -o -path "*/*/*/no_ps_bidir" -o -path "*/*/*/no_ps_onedir" | while read -r target_dir; do
  echo "Copying files to $target_dir"
  cp "$source_dir"/*.{py,txt,nc,jb} "$target_dir"/ 2>/dev/null || echo "  No matching files found in source directory"
  # cp "$source_dir"/*.{py,base,jb} "$target_dir"/ 2>/dev/null || echo "  No matching files found in source directory"
done

echo "File copying complete!"
