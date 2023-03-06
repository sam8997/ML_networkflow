#!/bin/bash

SECONDS=0

if [[ -z $1 ]]; then
    echo "Usage: ./run.sh <TARGET>"
    exit 1
else
    export INPUT_DIR="raw/${1#raw/}"
    export OUTPUT_DIR="csv/${1#raw/}"

    echo "Input directory: ${INPUT_DIR}"
    echo "Output directory: ${OUTPUT_DIR}"

    # Check output directory
    if [ -f ${OUTPUT_DIR} ]; then
        read -p "Output directory exists, remove? (Y/n)" selection
        if [[ $selection == 'n' ]]; then
            exit 1
        else
            rm -rf "${OUTPUT_DIR}"
            mkdir -p "${OUTPUT_DIR}"
            echo "Output directory removed and re-created"
        fi
    fi

    # Check if zeek is up
    docker container ls | grep zeek > /dev/null
    if [[ $? -ne 0 ]]; then
        echo "Zeek is not up, delete zeek and restart"
        docker rm zeek
        docker run --name zeek -d -it -v $(pwd):/pcap broplatform/bro:4.0.2 /bin/bash
        docker start zeek
    fi
fi

# Parse pcap to network flow with zeek
zeek () {
    echo "Parsing pcap to network flow";
    fullname="${1}"
    filename="$(basename $fullname)"
    base="${filename%.pcap}"
    origdir="$(dirname $fullname)"
    dstname="csv/${origdir#raw/}/$base"
    mkdir -p $dstname;

    docker exec -it zeek bash -c "\
        cd /pcap; \
        rm -f *.log; \
        zeek -r $fullname; \
        mv *.log $dstname; \
    "
}

# Transfers logs to feature representations
combine () {
    echo "Transfering logs to feature csv";
    fullname="${1}"
    filename="$(basename $fullname)"
    base="${filename%.pcap}"
    origdir="$(dirname $fullname)"
    dstname="csv/${origdir#raw/}/$base"

    echo "Combining $dstname";
    ./0-combine.py $dstname;
    ./1-preprocess.sh $dstname;
    ./2-statistic-tmux.sh $dstname;
    ./3-mergeall.sh $dstname

    if [[ -f "$origdir/IOC.txt" ]]; then
        echo "Found IOC.txt, data will be labeled";
        cp $origdir/IOC.txt $dstname;
        ./4-label.py $dstname
    fi
}

for i in $(ls ${INPUT_DIR}/*.pcap); do
    time zeek $i
    time combine $i
done


