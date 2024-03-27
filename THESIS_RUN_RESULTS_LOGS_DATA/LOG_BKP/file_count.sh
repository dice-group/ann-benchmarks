#!/bin/bash

# Iterate through directories recursively, starting from the current directory
find . -type d -readable -exec bash -c '
  # Get the directory path
  dir="$1"

  # Count the files within the directory, excluding hidden files
  file_count="$(ls -1UA "$dir" | grep -v / | wc -l)"

  # Print the directory name and file count, separated by a tab
  printf "%s\t%d\n" "$dir" "$file_count"
' sh {} \;

