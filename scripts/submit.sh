#!/bin/bash

# all files to submit will be put into run/submit
folder="run/submit_Cumple"
mkdir -p $folder
rm $folder/*

# build
python build.py 
python ./scripts/build.py -p -o release -j 40 -s
cp build/Cumple $folder/
cp build/partitionHyperGraph $folder/

# Other files
cp scripts/README.submit.md $folder/README.md
cp scripts/io_map.csv $folder/
cp scripts/io_map.cxx $folder/
cp scripts/io_map $folder/
cp scripts/macroplacement_bookshelf2vivado.py $folder/
