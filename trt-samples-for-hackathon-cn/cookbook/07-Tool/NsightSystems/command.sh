#!/bin/bash

rm -rf ./*.plan
nsys profile --force-overwrite=true -o model-BuildAndRun python3 main.py
./opt/nvidia/nsight-systems/2023.4.1/bin/nsys profile --force-overwrite=true -t cuda,nvtx,osrt -o model-OnlyRun     python3 main.py
