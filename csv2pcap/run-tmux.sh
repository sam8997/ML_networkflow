#!/bin/bash

filename="${1}"
mal_file="${2}"
output="${3}"

if [[ ! -e $filename ]]; then
    echo "Empty filename";
    exit
fi

if [[ ! -e $mal_file ]]; then
    echo "Empty malicious record"
    exit
fi

if [[ ! -e $output ]]; then
    output="result.pcap"
fi

split_size=$(dc -e "$(ls -l $filename | cut -d ' ' -f 5) 1000000 / 16 / 1 +p")
echo "Splitting, size: $split_size"
tcpdump -r $filename -w split -C $split_size 2>/dev/null
mv split split0

total_entry=$(ls -l split* | wc -l)
echo "Total entry: $total_entry"

echo "Extracting malicious packets"
# generate commands
full_cmd="tmux new-window \"python filter.py 0 $mal_file; tmux wait-for -S 0-label-done \" \; "
for i in $(seq 1 $((total_entry-1)) ); do
    full_cmd="$full_cmd new-window \"python filter.py $i $mal_file; tmux wait-for -S $i-label-done \" \; "
done

eval $full_cmd;
for i in $(seq 0 $((total_entry-1)) ); do
    tmux wait-for $i-label-done;
done

mergecap -w $output labeled-[0-9]*
echo "Done, file saved at $(readlink -f $output)"


# Cleanup
rm labeled* split*

