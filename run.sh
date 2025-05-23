#!/bin/bash
input_file=$1
output_dir=$2

echo "Input file: $input_file"
echo "Output dir: $output_dir"

python3 test.py "$input_file" "$output_dir"