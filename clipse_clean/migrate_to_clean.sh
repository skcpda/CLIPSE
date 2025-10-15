#!/bin/bash
# Migration script to move working files from old structure to clean structure

echo "=== MIGRATING TO CLEAN STRUCTURE ==="

# Create clean directory structure
mkdir -p src/{losses,data,utils,samplers}
mkdir -p scripts configs models results logs docs

# Copy working core files
cp ../src/train_clipse.py src/
cp ../src/eval_retrieval.py src/
cp ../src/datasets.py src/
cp -r ../src/losses src/
cp -r ../src/data src/
cp -r ../src/utils src/
cp -r ../src/samplers src/

# Copy working SLURM scripts
cp ../scripts/train_sanw_debias.sbatch scripts/
cp ../scripts/train_sanw_bandpass.sbatch scripts/
cp ../scripts/eval_retrieval.sbatch scripts/

# Copy working configs
cp ../configs/flickr8k_vitb32_*.yaml configs/

# Copy models if they exist
if [ -d "../models" ]; then
    cp -r ../models .
fi

# Copy results if they exist
if [ -d "../results" ]; then
    cp -r ../results .
fi

# Copy logs if they exist
if [ -d "../logs" ]; then
    cp -r ../logs .
fi

echo "=== MIGRATION COMPLETE ==="
echo "Clean structure created with working files"
echo "Legacy files remain in parent directory"
