# sh jsonl.sh gold_sent.txt gold_id.txt gold.jsonl

# rm cached_test_BertTokenizer_512 cached_test_BertTokenizer_512.lock
# mv seqlab_final/test_predictions.txt ./test.txt
# python helper/heuristics.py id.txt test.txt result.jsonl
# rm test.txt id.txt



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


SCRIPT_PATH_HEURISTICS="utils/ReviewAdvisor_modified/tagger/helper/heuristics.py"

for subset in ${SUBSET[@]}; do

    DATA_DIR=$DATA_ROOT"/$subset/reviews_aspectAnnotated/"
    
    dirs=("$DATA_DIR"/*/)
    total=${#dirs[@]}
    count=0

    echo "Found $total folders to process."

    for dir in "${dirs[@]}"; do

        dirname="${dir%/}"  # remove trailing slash
        ((count++))

        python $SCRIPT_PATH_HEURISTICS \
        "$dir/id.txt" "$dir/test_predictions.txt" "$DATA_DIR/result.jsonl"

        progress=$((count * 100 / total))
        printf "\rProgress: %3d%% (%d/%d) - Processing: %s" "$progress" "$count" "$total" "$(basename "$dirname")"

        rm ${dir}

    done
    echo -e "Done preparing $subset data."
done
echo "All done!"
