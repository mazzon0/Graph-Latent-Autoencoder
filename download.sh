#!/bin/bash
# Usage: ./download.sh [coco|clevr|all]   (default: coco)

DATASET="${1:-coco}"

download_coco() {
    BASE_DIR="data/datasets/coco"
    mkdir -p "$BASE_DIR/annotations"
    pushd "$BASE_DIR" > /dev/null || exit

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

    popd > /dev/null
    echo "--- COCO Setup Complete: $BASE_DIR ---"
}

download_clevr() {
    BASE_DIR="data/datasets/clevr"
    mkdir -p "$BASE_DIR"
    pushd "$BASE_DIR" > /dev/null || exit

    echo "--- Downloading CLEVR v1.0 (about 19 GB) ---"
    wget -c https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip
    unzip -q CLEVR_v1.0.zip 'CLEVR_v1.0/images/train/*' 'CLEVR_v1.0/images/val/*' 'CLEVR_v1.0/scenes/*'
    rm CLEVR_v1.0.zip

    popd > /dev/null
    echo "--- CLEVR Setup Complete: $BASE_DIR/CLEVR_v1.0 ---"
}

case "$DATASET" in
    coco)  download_coco ;;
    clevr) download_clevr ;;
    all)   download_coco; download_clevr ;;
    *)     echo "Unknown dataset '$DATASET'. Usage: ./download.sh [coco|clevr|all]"; exit 1 ;;
esac
