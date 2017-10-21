#!/bin/bash

#block(name=inference, threads=1, memory=5000, subtasks=1, gpu=true, hours=100)
    python -u main.py # python will print out immediatly rather than store in buffer first.
    echo "Done" 

# if you want to schedule multiple gpu jobs on a server, better to use this tool.
# run: `bash ./qsub-SurfaceNet_inference.sh`
# for installation & usage, please refer to the author's github: https://github.com/alexanderrichard/queueing-tool
