#!/bin/bash

# create dataset folders;
mkdir -p dataset/ABC_CHUNK_SC; mkdir dataset/ABC_CHUNK; mkdir dataset/ABC_CHUNK_VAL
# For Shape Completion
cd dataset/ABC_CHUNK_SC/
echo "retrieving ABC chunk for scene completion training..."
wget https://archive.nyu.edu/rest/bitstreams/89388/retrieve --no-check-certificate
echo "extracting ABC chunk for scene completion training..."
7z x retrieve -r -aou
# For SnapNet Training
cd ../ABC_CHUNK
echo "retrieving ABC chunk for SnapNet training..."
wget https://archive.nyu.edu/rest/bitstreams/89314/retrieve --no-check-certificate
echo "extracting ABC chunk for SnapNet training..."
7z x retrieve -r -aou
# For SnapNet Validation
cd ../ABC_CHUNK_VAL/
echo "retrieving ABC chunk for SnapNet testing..."
wget https://archive.nyu.edu/rest/bitstreams/89266/retrieve --no-check-certificate
echo "extracting ABC chunk for SnapNet testing..."
7z x retrieve -r -aou

cd ..