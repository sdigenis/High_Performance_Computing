#!/bin/bash

source /opt/intel/parallel_studio_xe_2020/psxevars.sh intel64
./seq_main -o -b -n 2000 -t 0.001 -i Image_data/texture17695.bin
