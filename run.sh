!#!/bin/bash
# TIRA will call this with:
# ./run.sh <input-directory> <output-directory>
#input directory should have both training and test set in it
pip3 install -r requirements.txt

python3 train.py "$1" "$2" 