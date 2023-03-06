#!/bin/bash
base="${1}"
orig_pwd="$pwd"
cd $base;

rm -f split*;

# Calculate split portion
total_entry=$(( $(wc data.csv -l | cut -d ' ' -f 1) / 30000 ))
total_entry=$(( total_entry + 1 ))
total_entry=$(( total_entry < 16 ? total_entry : 16 ))
echo $total_entry > total_entry.txt

split -n l/$total_entry --numeric-suffixes data.csv split_;
for i in $(ls split*); do mv $i $i.csv; done

header="$(head -n 1 split_00.csv)"
for i in $(ls split*); do
    echo "$header" > new_$i;
    cat $i >> new_$i;
    mv new_$i $i;
done
# Remove duplicated first line of split_00.csv
sed -i -e "1d" split_00.csv

cd $orig_pwd;
