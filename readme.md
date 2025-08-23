# Breaking the Reviewer: Assessing the Vulnerability of Large Language Models in Automated Peer Review Under Textual Adversarial Attacks

![Pipeline](pipeline.png)


*📝 Accepted at EMNLP 2025 Findings*

This repository contains the implementation for our paper that investigates the vulnerability of Large Language Models (LLMs) in automated peer review systems when subjected to textual adversarial attacks.

> **Note**: This repository is under active development and will be updated continuously.

## Reference Repositories

This project builds upon the following repositories:
- [TextAttack](https://github.com/QData/TextAttack) - Framework for adversarial attacks on NLP models
- [PromptBench](https://github.com/microsoft/promptbench.git) - Benchmark for prompt robustness

## Dataset Setup

### PeerRead Dataset

Clone the PeerRead repository:
```bash
git clone https://github.com/allenai/PeerRead.git data/dataset
```

Generate aspect score annotations for ICLR 2017 data:
```bash
python data/dataset/PeerRead/code/aspect_predict/assign_annot_iclr_2017.py
```

### AgentReview Dataset

Access the dataset from: https://github.com/Ahren09/AgentReview

### Dataset Aspect Tagging

This project uses the ReviewAdvisor tagger to add aspect tags to datasets.

#### Setup and Usage

1. Download the tagger and install dependencies:
```bash
bash utils/ReviewAdvisor_modified/download_tagger.sh
pip install transformers==3.0.2
pip install fire
```

2. Configure the tagging script by specifying the following parameters in `tag_dataset.sh`:
   - `<DATASET>` - Dataset name
   - `<DATA_ROOT>` - Root directory for data
   - `<DATASET_DIR>` - Dataset directory path
   - `<OUTPUT_DIR>` - Output directory for tagged data

3. Run the tagging process:
```bash
bash utils/ReviewAdvisor_modified/tag_dataset.sh
```

4. Restore transformers version:
```bash
pip install transformers==4.44.2
```

**Acknowledgements**: This aspect tagging implementation incorporates code adapted from [ReviewAdvisor](https://github.com/neulab/ReviewAdvisor). We have made certain modifications to fit our experimental setting, while preserving the original README and LICENSE files from the ReviewAdvisor repository. We gratefully acknowledge the authors of ReviewAdvisor for their valuable contribution.

