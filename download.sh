#!/bin/bash

BASE_DIR="data/datasets/coco"
mkdir -p "$BASE_DIR/annotations"
cd "$BASE_DIR" || exit

# Download and Extract Annotations
echo "--- Downloading COCO Annotations ---"
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip -j annotations_trainval2017.zip "annotations/instances_train2017.json" "annotations/instances_val2017.json" -d ./annotations/
rm annotations_trainval2017.zip

# Download and Extract Train 2017
echo "--- Downloading Train 2017 Images ---"
wget -c http://images.cocodataset.org/zips/train2017.zip
unzip -q train2017.zip
rm train2017.zip

# Download and Extract Val 2017
echo "--- Downloading Val 2017 Images ---"
wget -c http://images.cocodataset.org/zips/val2017.zip
unzip -q val2017.zip
rm val2017.zip

echo "--- Setup Complete ---"
echo "Structure created at $(pwd):"
ls -R
