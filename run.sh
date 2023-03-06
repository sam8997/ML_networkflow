#!/bin/bash

pcap="$(readlink -f ${1} 2>/dev/null)"
base=$(pwd)
model="${2}"

if [[ ! -e $pcap ]]; then
    echo -e "Empty pcap filename"
    echo -e "Usage: ./run.sh filename"
    exit 1
else
    basename=$(basename $pcap)
    name=${basename%.*}
fi

if [[ ! -e $model ]]; then
    model="lstm.h5";
fi

# Cleanup tmux before doing everything
tmux kill-server 2>/dev/null
tmux new-session -d

# Feature extraction
echo -e "\nExtracting feature"
rm -rf feature_extraction/raw/run_tmp/;
mkdir -p feature_extraction/raw/run_tmp/;
cp $pcap feature_extraction/raw/run_tmp/$basename;
cd feature_extraction
time ./run.sh raw/run_tmp
cd ..
rm -rf feature_extraction/raw/run_tmp


# Feed into model
echo -e "Classifying with model $model"
cd model
time pipenv run python detect.py --action predict --predict_dir $base/feature_extraction/csv/run_tmp/$name -o $name.csv -m $model
cd ..

# Map result back to pcap
echo -e "\nMapping result to pcap"
cd csv2pcap
time ./run-tmux.sh $pcap $base/model/$name.csv
cd ..
