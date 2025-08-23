from torch.utils.data import DataLoader

from .AgentReview_dataset import AgentReviewDataset, collate_fn_agentreview
from .PeerRead_dataset import ICLR2017_Dataset, collate_fn_iclr2017

DATASETS = {
    # PeerRead dataset
    "PeerRead": {
        "iclr_2017": {"dataset": ICLR2017_Dataset, "collate_fn": collate_fn_iclr2017},
    },
    # AgentReview dataset
    "AgentReview": {
        "iclr": {"dataset": AgentReviewDataset, "collate_fn": collate_fn_agentreview},
    },
}


def load_dataset(config):
    config = vars(config)

    # check dataset source
    if "PeerRead" in config["dataset_dir"]:
        dataset_source = "PeerRead"
    elif "AgentReview" in config["dataset_dir"]:
        dataset_source = "AgentReview"
    else:
        raise ValueError(f"Dataset source not found in {config['dataset_dir']}")

    # create dataset
    dataset = DATASETS[dataset_source][config["dataset_name"].lower()]["dataset"](
        **config
    )

    collate_fn = DATASETS[dataset_source][config["dataset_name"].lower()]["collate_fn"](
        config["input_metainfo"], config["input_contents"]
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=config["shuffle"],
        collate_fn=collate_fn,
    )
    return dataset, dataloader
