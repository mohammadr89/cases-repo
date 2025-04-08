#!/bin/bash

# Base reference directory
ref_base_dir="/home/mrashidi1/reference_codes"

# Find all instances of each compensation point directory type and copy the corresponding files
echo "Copying files to bi_dir directories..."
find . -type d -name "bi_dir" | while read -r target_dir; do
  echo "  Copying to $target_dir"
  cp "$ref_base_dir/bi_dir"/* "$target_dir"/ 2>/dev/null || echo "    No files found in reference bi_dir"
done

echo "Copying files to one_dir directories..."
find . -type d -name "one_dir" | while read -r target_dir; do
  echo "  Copying to $target_dir"
  cp "$ref_base_dir/one_dir"/* "$target_dir"/ 2>/dev/null || echo "    No files found in reference one_dir"
done

echo "Copying files to no_ps_bidir directories..."
find . -type d -name "no_ps_bidir" | while read -r target_dir; do
  echo "  Copying to $target_dir"
  cp "$ref_base_dir/no_ps_bidir"/* "$target_dir"/ 2>/dev/null || echo "    No files found in reference no_ps_bidir"
done

echo "Copying files to no_ps_onedir directories..."
find . -type d -name "no_ps_onedir" | while read -r target_dir; do
  echo "  Copying to $target_dir"
  cp "$ref_base_dir/no_ps_onedir"/* "$target_dir"/ 2>/dev/null || echo "    No files found in reference no_ps_onedir"
done

echo "File copying complete!"
