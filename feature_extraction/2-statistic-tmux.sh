#!/bin/bash
dir="${1}"
total_entry=$(cat $dir/total_entry.txt)

# generate commands
full_cmd="tmux new-window \"python statistic.py "$dir" 0; tmux wait-for -S 0-done \" \; "
for i in $(seq 1 $((total_entry-1)) ); do
    full_cmd="$full_cmd new-window \"python statistic.py "$dir" $i; tmux wait-for -S $i-done \" \; "
done

eval $full_cmd;

for i in $(seq 0 $((total_entry-1)) ); do
    tmux wait-for $i-done;
done
