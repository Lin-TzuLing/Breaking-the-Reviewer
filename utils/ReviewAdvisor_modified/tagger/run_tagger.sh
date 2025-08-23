#!/bin/bash

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


SCRIPT_PATH="utils/ReviewAdvisor_modified/tagger/run_tagger.py"

for subset in ${SUBSET[@]}; do

    DATA_DIR=$DATA_ROOT"/$subset/reviews_aspectAnnotated"
    
    dirs=("$DATA_DIR"/*/)
    total=${#dirs[@]}
    count=0

    echo "Found $total folders to process."

    for dir in "${dirs[@]}"; do

        dirname="${dir%/}"  
        ((count++))

    
        TF_CPP_MIN_LOG_LEVEL=3 TF_ENABLE_ONEDNN_OPTS=0\
        python utils/ReviewAdvisor_modified/tagger/run_tagger.py \
            --data_dir "$dir" \
            --labels "utils/ReviewAdvisor_modified/tagger/labels.txt" \
            --model_name_or_path "utils/ReviewAdvisor_modified/tagger/seqlab_final" \
            --evaluate_during_training \
            --output_dir "$dir" \
            --max_seq_length 512 \
            --num_train_epochs 3 \
            --per_device_train_batch_size 2 \
            --per_device_eval_batch_size 6 \
            --save_steps 20680 \
            --eval_steps 20680 \
            --seed 1 \
            --do_predict \
        2>/dev/null

        rm "${dir}cached_test_BertTokenizer_512" "${dir}cached_test_BertTokenizer_512.lock"
        rm "${dir}test_results.txt" "${dir}test.txt"

        # Progress bar
        progress=$((count * 100 / total))
        printf "\rProgress: %3d%% (%d/%d) - Processing: %s" "$progress" "$count" "$total" "$(basename "$dirname")"
    done
    echo -e "Done preparing $subset data."
done
echo "All done!"

# python utils/ReviewAdvisor_modified/tagger/run_tagger.py \
#     --data_dir "$dir" \
#     --labels "utils/ReviewAdvisor_modified/tagger/labels.txt" \
#     --model_name_or_path "utils/ReviewAdvisor_modified/tagger/seqlab_final" \
#     --evaluate_during_training true \
#     --output_dir "ttt" \
#     --max_seq_length 512 \
#     --num_train_epochs 3 \
#     --per_device_train_batch_size 2 \
#     --per_device_eval_batch_size 6 \
#     --save_steps 20680 \
#     --eval_steps 20680 \
#     --seed 1 \
#     --do_predict 