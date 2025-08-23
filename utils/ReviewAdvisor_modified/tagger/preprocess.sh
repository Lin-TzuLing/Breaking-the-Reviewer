#!/bin/bash


# DATASET="PeerRead"
DATASET="AgentReview"

if [ "$DATASET" = "PeerRead" ]; then
    SUBSET=("train" "dev" "test")
    DATASET_ROOT="data/dataset/PeerRead/data/iclr_2017"
    REVIEW_DIR="reviews"
elif [ "$DATASET" = "AgentReview" ]; then
    SUBSET=('ICLR2020' 'ICLR2021' 'ICLR2022' 'ICLR2023')
    DATASET_ROOT="data/dataset/AgentReview"
    REVIEW_DIR="notes"
else
    echo "Invalid dataset specified. Please set DATASET to either 'PeerRead' or 'AgentReview'."
    exit 1
fi

# ("Usage: python tagger_preprocess.py <dataset_name> <dataset_dir> <output_dir>")
SCRIPT_PATH="utils/ReviewAdvisor_modified/tagger/tagger_preprocess.py"

for subset in ${SUBSET[@]}; do
    count=0

    DATASET_DIR="${DATASET_ROOT}/${subset}/${REVIEW_DIR}"
    OUTPUT_DIR="${DATASET_ROOT}/${subset}/reviews_aspectAnnotated/"

    # create directory if not exist
    if [ ! -d "$OUTPUT_DIR" ]; then 
        mkdir -p "$OUTPUT_DIR"
    fi

    echo "Preprocessing '$DATASET_DIR' reviews to '$OUTPUT_DIR' ..."

    python $SCRIPT_PATH $DATASET $DATASET_DIR $OUTPUT_DIR

    echo "Done preprocessing $subset data."
    break
done
