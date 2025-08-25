#!/bin/bash

export TF_CPP_MIN_LOG_LEVEL=3

# PeerRead Dataset Examples
echo "=== PeerRead Dataset ==="

# OpenAI Models
echo "Running attacks on GPT-4o-mini..."
python main.py --attack DeepWordBug --manual_review_root 'data/clean_review/API/PeerRead_iclr_2017/gpt-4o-mini' --model_name 'gpt-4o-mini'

echo "Running attacks on GPT-4o..."
python main.py --attack DeepWordBug --manual_review_root 'data/clean_review/API/PeerRead_iclr_2017/gpt-4o' --model_name 'gpt-4o'

# Local Models (requires model server running)
echo "Running attacks on Llama-3.3-70B..."
python main.py --attack DeepWordBug --manual_review_root 'data/clean_review/API/PeerRead_iclr_2017/Llama-3.3-70B' --model_name 'Llama-3.3-70B' --port 8091

# AgentReview Dataset Examples
echo "=== AgentReview Dataset ==="

echo "Running attacks on GPT-4o-mini with AgentReview..."
python main.py --attack DeepWordBug --dataset_name 'iclr' --dataset_dir 'data/dataset/AgentReview' --manual_review_root 'data/clean_review/API/AgentReview/gpt-4o-mini' --model_name 'gpt-4o-mini'

echo "Running attacks on Llama-3.3-70B with AgentReview..."
python main.py --attack Styleadv --dataset_name 'iclr' --dataset_dir 'data/dataset/AgentReview' --manual_review_root 'data/clean_review/API/AgentReview/Llama-3.3-70B' --model_name 'Llama-3.3-70B' --port 8091

echo "All attacks completed!"