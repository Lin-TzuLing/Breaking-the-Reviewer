#!/bin/bash


# Dataset name for review aspect annotation
DATASET="PeerRead"
# DATASET="AgentReview"

if [ "$DATASET" = "PeerRead" ]; then
    SUBSET=("train" "dev" "test")
    DATA_ROOT="data/dataset/PeerRead/data/iclr_2017"
    REVIEW_DIR="reviews"
elif [ "$DATASET" = "AgentReview" ]; then
    SUBSET=('ICLR2020' 'ICLR2021' 'ICLR2022' 'ICLR2023')
    DATA_ROOT="data/dataset/AgentReview"
    REVIEW_DIR="notes"
else
    echo "Invalid dataset specified. Please set DATASET to either 'PeerRead' or 'AgentReview'."
    exit 1
fi

# Set Script paths
SCRIPT_PREPROCESS="utils/ReviewAdvisor_modified/tagger/tagger_preprocess.py" # step 1
SCRIPT_JSONLIZE="utils/ReviewAdvisor_modified/tagger/helper/jsonlize.py" # step 2
SCRIPT_SPLIT="utils/ReviewAdvisor_modified/tagger/helper/split.py" # step 2
SCRIPT_TAGGER="utils/ReviewAdvisor_modified/tagger/run_tagger.py" # step 3
SCRIPT_HEURISTICS="utils/ReviewAdvisor_modified/tagger/helper/heuristics.py" # step 4
LABELS_PATH="utils/ReviewAdvisor_modified/tagger/labels.txt" # step 3
MODEL_PATH="utils/ReviewAdvisor_modified/tagger/seqlab_final" # step 3

# Download nltk.punkt Module. Change transformers version to 3.0.2 for aspect tagger
python utils/ReviewAdvisor_modified/tagger/helper/download_punkt.py
pip install transformers==3.0.2
pip install fire


echo "*** Processing Dataset: $DATASET ***"

for subset in "${SUBSET[@]}"; do
    echo "=== Processing subset: $subset ==="

    # Step 1. Preprocess reviews (gather reviews from parsed dataset) 
    DATASET_DIR="${DATA_ROOT}/${subset}/${REVIEW_DIR}"
    OUTPUT_DIR="${DATA_ROOT}/${subset}/reviews_aspectAnnotated/"

    mkdir -p "$OUTPUT_DIR"
    echo "[1/4] Preprocess reviews -> $OUTPUT_DIR"
    python $SCRIPT_PREPROCESS $DATASET $DATASET_DIR $OUTPUT_DIR

    # Step 2. Process txt files to jsonl and split into test.txt and id.txt
    files=("$OUTPUT_DIR"/*.txt)
    total_files=${#files[@]}
    echo "[2/4] Process txt files to jsonl and split into test.txt and id.txt ($total_files files)"

    count=0
    for file in "${files[@]}"; do
        filename=$(basename "${file%.*}")
        RESULT_DIR="$OUTPUT_DIR/$filename"
        mkdir -p "$RESULT_DIR"

        out_jsonl="$RESULT_DIR/out.jsonl"
        out_test_txt="$RESULT_DIR/test.txt"
        out_id_txt="$RESULT_DIR/id.txt"

        python $SCRIPT_JSONLIZE "$file" "$out_jsonl"

        TF_CPP_MIN_LOG_LEVEL=3 TF_ENABLE_ONEDNN_OPTS=0\
        python $SCRIPT_SPLIT "$out_jsonl" 1 "$out_test_txt" "$out_id_txt" 2>/dev/null
        
        rm "$file" "$out_jsonl" 

        count=$((count + 1))
        progress=$((100 * count / total_files))
        printf "\rProgress: %3d%% (%d/%d files)" "$progress" "$count" "$total_files"
        
    done
    echo
    echo "Done txt preprocess of ($subset) subset"

    # Step 3. Run tagger model to predict aspect annotations for reviews
    echo "[3/4] Run tagger model to predict aspect annotations for reviews"

    dirs=("${OUTPUT_DIR}"*/)
    total_dirs=${#dirs[@]}
    count=0


    for dir in "${dirs[@]}"; do

        dirname="${dir%/}"  
        ((count++))

    
        TF_CPP_MIN_LOG_LEVEL=3 TF_ENABLE_ONEDNN_OPTS=0\
        python $SCRIPT_TAGGER \
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

        rm -f "${dir}cached_test_BertTokenizer_512" "${dir}cached_test_BertTokenizer_512.lock"
        rm -f "${dir}test_results.txt" "${dir}test.txt"

        progress=$((count * 100 / total_dirs))
        printf "\rProgress: %3d%% (%d/%d files)" "$progress" "$count" "$total_dirs"
        
    done
    echo
    echo "Done tag prediction of ($subset) subset"

    # Step 4. Merge predictions into jsonl
    echo "[4/4] Merge prediction into jsonl"

    count=0
    for dir in "${dirs[@]}"; do
        dirname="${dir%/}"
        paper_id=$(basename "$dirname")
        ((count++))

        python $SCRIPT_HEURISTICS "$dir/id.txt" "$dir/test_predictions.txt" "$OUTPUT_DIR/$paper_id.jsonl"

        rm -rf "${dir}"

        progress=$((count * 100 / total_dirs))
        printf "\rProgress: %3d%% (%d/%d files)" "$progress" "$count" "$total_dirs"
        
    done
    echo 
    echo "Done heuristics process of ($subset) subset"
    echo "=== Done Processing subset: $subset ==="
    
done

echo "*** Done Processing dataset: $DATASET (for reviews aspect annotation) ***"

# Re-install transformers to 4.44.2 for the main project
pip install transformers==4.44.2