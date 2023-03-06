#!/bin/bash
dir="${1}"
total_entry=$(cat $dir/total_entry.txt)

cp "$dir/ct_00.csv" "$dir/full_data.csv"
for i in $(seq 1 $((total_entry-1)) ); do
    name="$dir/ct_$(printf '%02d' $i).csv"
    echo "Appending $name"
    tail -n +2 $name >> "$dir/full_data.csv"
done
