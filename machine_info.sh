#!/bin/bash

# CPU
echo "=================CPU================"
lscpu | grep "Model name"


# Motherboard
echo "============Motherboard============="
sudo dmidecode -t 2

# Hard Disk
echo "============Motherboard============="
sudo lshw | grep "TS512GMTE110S"

# Memory
echo "============Memory=================="
sudo dmidecode --type memory

# Graphics card
echo "============Graphics  card=========="
nvidia-smi
