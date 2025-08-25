import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
import datetime
import json
import logging
import random
import warnings

import flair
import numpy as np
import tensorflow as tf
import torch

from attack.attack import Attack
from data import load_dataset
from model import load_model
from prompt import PromptTemplate

logging.getLogger("tensorflow").setLevel(logging.FATAL)
flair.logger.setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
tf.get_logger().setLevel("ERROR")
gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    # Dataset
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="iclr_2017",
        help="Name of the dataset.",
        # "iclr" for AgentReview
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="data/dataset/PeerRead/data/iclr_2017",
        help="Directory containing the dataset.",
        # "data/dataset/AgentReview" for AgentReview
    )
    parser.add_argument(
        "--dataset_mode",
        type=str,
        default="all",
        choices=["all", "train", "dev", "test"],
        help="Mode of the dataset.",
        # ["all", "ICLR_2020", "ICLR_2021", "ICLR_2022", "ICLR_2023"] for AgentReview
    )
    parser.add_argument(
        "--manual_review_root",
        type=str,
        help="Root directory containing manual reviews.",
    )

    # Data
    parser.add_argument(
        "--input_metainfo",
        nargs="+",
        default=["paper_id"],
        help="meta information to be included",
    )
    parser.add_argument(
        "--input_contents",
        nargs="+",
        default=[],
        help="Input content for each paper (to be reviewed).",
    )
    parser.add_argument(
        "--modifiable_contents",
        nargs="+",
        default=["lcs"],
        help="Contents (or sections) that are modifiable (inject attack patterns into) for each paper.",
    )
    parser.add_argument(
        "--aspect_score_types",
        nargs="+",
        default=[
            "OVERALL",
            "SUBSTANCE",
            "APPROPRIATENESS",
            "MEANINGFUL COMPARISON",
            "SOUNDNESS CORRECTNESS",
            "ORIGINALITY",
            "CLARITY",
            "IMPACT",
        ],
        help="Aspect for generating score.",
    )
    parser.add_argument(
        "--aspect_tag_types",
        nargs="+",
        default=[
            "NONE",
            "SUMMARY",
            "MOTIVATION POSITIVE",
            "MOTIVATION NEGATIVE",
            "SUBSTANCE POSITIVE",
            "SUBSTANCE NEGATIVE",
            "ORIGINALITY POSITIVE",
            "ORIGINALITY NEGATIVE",
            "SOUNDNESS POSITIVE",
            "SOUNDNESS NEGATIVE",
            "CLARITY POSITIVE",
            "CLARITY NEGATIVE",
            "REPLICABILITY POSITIVE",
            "REPLICABILITY NEGATIVE",
            "MEANINGFUL COMPARISON POSITIVE",
            "MEANINGFUL COMPARISON NEGATIVE",
        ],
        help="Aspect tags for generating review content.",
    )
    parser.add_argument(
        "--data_label_type", type=str, default="avg", choices=["avg", "most_feq"]
    )
    parser.add_argument("--port", type=int, default=8091, choices=[8091, 8092])

    # Model
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-4o-mini",
        choices=["gpt-4o-mini", "gpt-4o", "Llama-3.3-70B", "Mistral-small-3.1"],
        help="Model to use",
    )
    # change this to config
    parser.add_argument("--openai_key", type=str, default="")

    # Train & Test
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle the dataset")

    # Adversarial Attack
    parser.add_argument(
        "--attack",
        type=str,
        default="",
        choices=[
            "DeepWordBug",
            "PuncAttack",
            "TextFooler",
            "BertAttack",
            "StyleAdv",
        ],
        help="Adversarial attack to use",
    )

    # Output
    parser.add_argument(
        "--output_explanation",
        action="store_true",
        help="Output explanation while generating review",
    )

    # Output directory
    parser.add_argument(
        "--output_dir", type=str, default="result_EMNLP/", help="Output directory"
    )

    # resume
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume the attack, skip the attacked papers.",
    )
    parser.add_argument(
        "--run_paper_num",
        type=int,
        default=None,
        help="number of papers to run, None means all papers. including existing papers.",
    )
    args = parser.parse_args()
    return args


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def main():
    # Load arguments & set seed, output directory, openai key
    args = get_params()

    args.device = (
        "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    )
    print(f"Using device: {args.device}")
    set_seed(args.seed)

    parts = args.dataset_dir.split("/")
    filtered_datasetName = [part for part in parts if "data" not in part]
    filtered_datasetName = "_".join(filtered_datasetName)
    args.output_dir = os.path.join(
        args.output_dir, filtered_datasetName, args.model_name, args.attack
    )

    os.makedirs(args.output_dir, exist_ok=True)
    with open("openai_key.txt", "r") as f:
        args.openai_key = (
            f.readline().strip()
        )  # remember to set your openai key in openai_key.txt

    # Load dataset
    dataset, dataloader = load_dataset(args)

    # Load model
    model = load_model(args)
    print(f"Model : {model.__class__.__name__}")

    # Load prompter
    prompter = PromptTemplate(
        aspect_tag_types=args.aspect_tag_types,
        aspect_score_types=args.aspect_score_types,
        explain=args.output_explanation,
    )

    # Load attacker
    attacker = Attack(
        model=model, attack_name=args.attack, prompter=prompter, verbose=True
    )
    print(f"Attacker : {attacker.__class__.__name__}")
    print("=======================================================")

    # resume
    skip_id_list = dataset.skip_id_list

    output_list = []
    for idx, (data, label) in enumerate(dataloader):

        if args.run_paper_num is not None and (idx + 1) >= args.run_paper_num:
            break

        # Prepare input data
        paper_id = [d["paper_id"] for d in data]

        if (
            paper_id[0] not in dataset.manual_review.keys()
            or paper_id[0] in skip_id_list
        ):
            continue

        paper_content = [d["paper_content"] for d in data]

        # Load attack
        print(f"Attacking paper {paper_id}...")
        print(datetime.datetime.now())

        # Attack and get results
        try:
            result = attacker.attack(paper_content)

            # Save results
            for i in range(len(result)):
                entry = {
                    "paper_id": paper_id[i],
                    # input prompt (paper content)
                    "original_content": result[i]["original_content"],
                    "attacked_content": result[i]["attacked_content"],
                    # model output (generated review & score)
                    "original_output": result[i]["original_output"],
                    "attacked_output": result[i]["attacked_output"],
                    # total score of model output (sum of all aspect scores)
                    "original_score": result[i]["original_score"],
                    "attacked_score": result[i]["attacked_score"],
                    # ground truth score
                    "ground_truth": label[i],
                    # num of queries
                    "num_queries": result[i]["num_queries"],
                    # score increase (attacked_score - original_score)
                    "score_shift": result[i]["score_shift"],
                    # if attack is successful (score_shift > 0)
                    "attack_success": result[i]["attack_success"],
                }

                # verbose
                print(
                    f"=> Paper ID: {entry['paper_id']}, Attack Success: {entry['attack_success']}, Score Shift: {entry['score_shift']}"
                )
                print("=======================================================")
                output_list.append(entry)
                with open(
                    os.path.join(
                        args.output_dir,
                        f"{args.attack}_Explain{args.output_explanation}.jsonl",
                    ),
                    "a",
                ) as f:
                    f.write(json.dumps(entry) + "\n")
                # exit()
        except SystemExit:
            raise
        except Exception as e:
            print(f"Error: {paper_id}, Exception: {e}.")
            with open(
                os.path.join(
                    args.output_dir,
                    f"{args.attack}_Explain{args.output_explanation}_Error.jsonl",
                ),
                "a",
            ) as f:
                f.write(json.dumps({"paper_id": paper_id, "error": {e}}) + "\n")

    # Save output
    with open(os.path.join(args.output_dir, f"{args.attack}_total.jsonl"), "w") as f:
        # infomation
        f.write(
            json.dumps(
                {
                    "dataset": args.dataset_dir,
                    "dataset_mode": args.dataset_mode,
                    "input_contents": args.input_contents,
                    "modifiable_contents": args.modifiable_contents,
                    "output_explanation": args.output_explanation,
                    "model_name": args.model_name,
                    "attack": args.attack,
                    "manual_review_root": args.manual_review_root,
                }
            )
            + "\n"
        )
        # results
        for entry in output_list:
            f.write(json.dumps(entry) + "\n")

    print("Done.")


if __name__ == "__main__":
    main()
