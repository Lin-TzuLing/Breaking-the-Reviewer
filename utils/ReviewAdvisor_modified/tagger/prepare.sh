# revised, 2024/08
#!/bin/bash

python utils/ReviewAdvisor_modified/tagger/helper/download_punkt.py
# DATASET="PeerRead"
DATASET="AgentReview"

if [ "$DATASET" = "PeerRead" ]; then
    SUBSET=("train" "dev" "test")
    DATA_ROOT="data/dataset/PeerRead/data/iclr_2017"
elif [ "$DATASET" = "AgentReview" ]; then
    SUBSET=('ICLR2020' 'ICLR2021' 'ICLR2022' 'ICLR2023')
    DATA_ROOT="data/dataset/AgentReview"
else
    echo "Invalid dataset specified. Please set DATASET to either 'PeerRead' or 'AgentReview'."
    exit 1
fi

SCRIPT_PATH_JSONLIZE="utils/ReviewAdvisor_modified/tagger/helper/jsonlize.py"
SCRIPT_PATH_SPLIT="utils/ReviewAdvisor_modified/tagger/helper/split.py"

for subset in ${SUBSET[@]}; do
    count=0
    DATA_DIR="${DATA_ROOT}/${subset}/reviews_aspectAnnotated/"

    files=("$DATA_DIR"/*.txt)
    total_files=${#files[@]}
    echo "Processing $total_files files in subset $subset..."

    # loop through all files in the directory and get reviews tagged
    for file in "${files[@]}"; do

        filename=$(basename "${file%.*}")

        RESULT_DIR="${DATA_ROOT}/${subset}/reviews_aspectAnnotated/${filename}/"  
        # create directory if not exist
        if [ ! -d "$RESULT_DIR" ]; then 
            mkdir -p "$RESULT_DIR"
        fi

        out_jsonl=$RESULT_DIR/"out.jsonl"
        out_test_txt=$RESULT_DIR/"test.txt"
        out_id_txt=$RESULT_DIR/"id.txt"

        python $SCRIPT_PATH_JSONLIZE $file $out_jsonl 
        TF_CPP_MIN_LOG_LEVEL=3 TF_ENABLE_ONEDNN_OPTS=0 python $SCRIPT_PATH_SPLIT $out_jsonl 1 $out_test_txt $out_id_txt 2>/dev/null

        if [ $? -ne 0 ]; then
            echo $file
            echo " Error occurred in Python script for $file, exiting loop."
            break
        fi
        rm $out_jsonl
        rm $file

        count=$((count + 1))
        progress=$((100 * count / total_files))
        printf "\rProgress: %3d%% (%d/%d files)" "$progress" "$count" "$total_files"
    done
    echo "Done preparing $subset data."
done


# sh prepare.sh results/review.test gold_sent.txt gold_id.txt
# python helper/jsonlize.py "$1" out.jsonl
# python helper/split.py out.jsonl 1 test.txt id.txt
# rm out.jsonl