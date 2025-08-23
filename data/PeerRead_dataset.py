import json
import os
import re
import warnings
from typing import Optional

import numpy as np
from torch.utils.data import Dataset

from data.utils.lcs_match import add_modifiable_markers
from prompt import PromptTemplate

warnings.filterwarnings("ignore")


def collate_fn_iclr2017(input_metainfo, input_contents):
    def collate_fn(batch):
        x, y = [], []
        for _, (data, label) in enumerate(batch):
            x.append(data)
            y.append(label)
        return (x, y)

    return collate_fn


class ICLR2017_Dataset(Dataset):
    """
    PeerRead ICLR 2017 dataset.

    Args:
        dataset_name (:obj:`str`, optional): the name of the dataset. Default to 'iclr_2017'.
        dataset_dir (:obj:`str`): the path to the raw dataset. Default to 'data/raw_dataset/PeerRead/data/iclr_2017'.
        dataset_mode (:obj:`str`): the split of data. choices = ['all', 'train', 'dev', 'test']. Default to 'all'.
        data_label_type (:obj:`str`): the score label type. choices = ['avg', 'most_feq']. Default to 'avg'.
    """

    def __init__(
        self,
        dataset_name: Optional[str] = "iclr_2017",
        dataset_dir: str = "data/dataset/PeerRead/data/iclr_2017",
        dataset_mode: str = "all",
        data_label_type: Optional[str] = "avg",
        **kwargs,
    ):
        self.path = dataset_dir
        self.mode = dataset_mode

        self.attack = kwargs.get("attack", None)
        self.input_metainfo = kwargs.get("input_metainfo", None)
        self.modifiable_contents = kwargs.get("modifiable_contents", None)
        self.input_contents = kwargs.get("input_contents", None)
        self.manual_review_root = kwargs.get("manual_review_root", None)
        self.resume = kwargs.get("resume", False)
        self.output_explanation = kwargs.get("output_explanation", False)
        self.output_dir = kwargs.get("output_dir", None)
        self.model_name = kwargs.get("model_name", None)

        self.re_index = 0
        self.paper_id_mapping = {}  # paper_id_mapping : {key: paper_id, value: re_index}

        # resume
        self.skip_id_list = []
        if self.resume:
            with open(
                os.path.join(
                    self.output_dir,
                    f"{self.attack}_Explain{self.output_explanation}.jsonl",
                ),
                "r",
            ) as f:
                for line in f:
                    data = json.loads(line)
                    paper_id = data["paper_id"]
                    self.skip_id_list.append(paper_id)

        print("=======================================================")
        print(f"Loading {dataset_name} dataset . . .")

        print(f"\n=> Reading {self.mode} data . . .")
        if self.mode in ["train", "dev", "test"]:  # single split
            self.datas = self.read_data(self.path + "/" + self.mode)
        elif self.mode == "all":  # all splits combined
            self.datas = {}
            for mode in ["train", "dev", "test"]:
                self.datas = {**self.datas, **self.read_data(self.path + "/" + mode)}
        else:
            raise ValueError(f"=> Invalid data mode {self.mode}")

        print("\n=> Computing aspect ground truth score . . .")
        self.min_score, self.max_score = float("inf"), float("-inf")
        self.label = self.compute_labels(data_label_type)

        # process read manual review for long common subsequence match
        self.manual_review, self.lcs_split_level = self.read_manual_review(
            self.manual_review_root
        )

        assert len(self.label) == len(self.datas)
        print(f"\n{dataset_name} dataset loaded successfully!")
        print("=======================================================")

    def read_data(self, root):
        """
        Read data from the raw dataset.

        Args:
            root (:obj:`str`): the path to the raw dataset.

        Returns:
            datas (:obj:`dict`): the data of all papers.
        """

        # read review files
        path_review = root + "/reviews"
        data_review = {}
        num_avg_reviews, num_min_reviews = 0, 100
        print(f"Reading review files from {path_review} . . .")
        for filename in sorted(os.listdir(path_review)):
            paper_id = filename.split(".")[0]
            with open(path_review + "/" + filename, "r", encoding="utf-8") as f:
                data = json.loads(f.read())
                # filter out empty comments
                comments = [
                    self._normalize_string(r["comments"]) for r in data["reviews"]
                ]
                comments = [c for c in comments if c != ""]
                data["review_comments_list"] = sorted(list(set(comments)))
                data_review[paper_id] = data

                if len(data_review[paper_id]["review_comments_list"]) < num_min_reviews:
                    num_min_reviews = len(data_review[paper_id]["review_comments_list"])
                num_avg_reviews += len(data_review[paper_id]["review_comments_list"])

        print(f"num of reviews : {len(data_review)}")
        print(f"avg reviews per paper: {(num_avg_reviews/len(data_review)):.4f}")
        print(f"min reviews per paper: {num_min_reviews}")

        # read paper files
        path_paper = root + "/parsed_pdfs"
        data_paper = {}
        for filename in sorted(os.listdir(path_paper)):
            paper_id = filename.split(".")[0]
            with open(path_paper + "/" + filename, "r", encoding="utf-8") as f:
                data_paper[paper_id] = json.loads(f.read())
        print(f"num of papers : {len(data_paper)}")

        assert data_review.keys() == data_paper.keys()

        # merge data
        datas = {}
        num_abstract = 0
        for id in data_paper.keys():
            review = data_review[id]
            paper = data_paper[id]
            assert review["title"] is not None
            assert id == review["id"]

            paper["metadata"] = {
                k: v
                for k, v in paper["metadata"].items()
                if k not in ["title", "authors", "abstractText"]
            }
            review["authors"] = review["authors"].split(", ")
            review["paper_id"] = review.pop("id")

            # abstract and sections
            paper_contents = {}
            paper_contents["abstract"] = review["abstract"]
            num_abstract += len(review["abstract"].split())
            if paper["metadata"]["sections"] is not None:
                for sec in paper["metadata"]["sections"]:
                    paper_contents[f"Section {sec['heading']}"] = sec["text"]

            paper["metadata"].pop("sections")
            paper["paper_contents"] = paper_contents
            review.pop("abstract")

            datas[self.re_index] = {**review, **paper}  # merge two dict
            self.paper_id_mapping[id] = self.re_index
            self.re_index += 1
        print(f"avg abstract length per paper : {num_abstract/len(data_paper)}")

        return datas

    def read_manual_review(self, manual_review_root):
        """
        Read manual review files from the specified directory.

        Args:
            manual_review_root (:obj:`str`): the root directory of manual review files.

        Returns:
            manual_review (:obj:`dict`): a dictionary containing manual reviews.
            lcs_split_level (:obj:`str`): the split level for LCS.
        """
        
        manual_review = {}
        lcs_split_level = None

        # read manual review
        if any(x in self.modifiable_contents for x in ("lcs", "LCS")):
            assert self.model_name.split("-")[0] in manual_review_root.split("/")[-1]
            files = os.listdir(manual_review_root)
            for file in files:
                path = os.path.join(manual_review_root, file)
                paper_idx = file.split(".")[0]
                with open(path, "r") as f:
                    content = f.read()
                    manual_review[paper_idx] = PromptTemplate.parseManualReview(content)

            print(f"num of manual review : {len(manual_review)}")

            char_level_attack = ["DeepWordBug", "PuncAttack"]
            word_level_attack = ["TextFooler", "BertAttack", "Checklist"]
            sent_level_attack = ["StyleAdv", "Syntacticadv"]
            if self.attack in char_level_attack:
                lcs_split_level = "char"
            elif self.attack in word_level_attack:
                lcs_split_level = "word"
            elif self.attack in sent_level_attack:
                lcs_split_level = "sent"
            else:
                raise ValueError(f"Invalid attack {self.attack}")

        return manual_review, lcs_split_level

    def process_modifiable_contents(self, datas):
        """
        Process modifiable contents from the data.

        Args:
            datas (:obj:`dict`): the data of the paper.

        Returns:
            modifiable_contents (:obj:`dict`): the modifiable contents of the paper.
        """

        # split paper content and meta information
        paper_id = datas["paper_id"]
        data_content = datas["paper_contents"]
        data_meta = {k: v for k, v in datas.items() if k != "paper_contents"}

        x = {}
        # add meta information, e.g. paper_id
        if self.input_metainfo:
            x.update({k: v for k, v in data_meta.items() if k in self.input_metainfo})
        else:
            x.append(data_meta)

        # add input paper content, e.g. abstract, sections
        paper_content = {}
        if self.input_contents:
            paper_content.update(
                {k: v for k, v in data_content.items() if k in self.input_contents}
            )
        else:
            paper_content.update(data_content)

        # clarify modifiable contents
        if self.modifiable_contents and not any(
            x in self.modifiable_contents for x in ("lcs", "LCS")
        ):
            for k, v in paper_content.copy().items():
                if k not in (self.modifiable_contents + self.input_metainfo):
                    new_k = "<UnmodifiableStart>" + k.capitalize()
                    new_v = v + "<UnmodifiableEnd>"
                    paper_content[new_k] = new_v
                    paper_content.pop(k)
                else:
                    new_k = k.capitalize()
                    paper_content[new_k] = v
                    paper_content.pop(k)
        else:
            # TODO: find LCS here
            content_str = ""
            for key, value in paper_content.items():
                if (
                    key not in self.input_metainfo
                ):  # do not include meta information except paper_id
                    content_str += f"{key.capitalize()}: {value}\n"

            # Longest Common Subsequence finding
            if (
                any(x in self.modifiable_contents for x in ("lcs", "LCS"))
                and paper_id in self.manual_review.keys()
            ):
                patterns = self.manual_review[paper_id][0].values()
                content = add_modifiable_markers(
                    content_str, patterns, self.lcs_split_level
                )
                x["paper_content"] = content
            else:
                x["paper_content"] = (
                    content_str  # no manual review, no unmodifiable markers
                )
                print(f"paper {paper_id} not in manual review")

        print(f"paper {paper_id} processed")
        return x

    def compute_labels(self, label_type):
        """
        Process aspect score label for each paper & example output template.

        Args:
            label_type (:obj:`str`): the score label type. choices = ['avg', 'most_feq'].

        Returns:
            aspect_label (:obj:`dict`): aspect label for all papers.
        """

        aspects = [
            "RECOMMENDATION",
            "RECOMMENDATION_UNOFFICIAL",
            "SUBSTANCE",
            "APPROPRIATENESS",
            "MEANINGFUL_COMPARISON",
            "SOUNDNESS_CORRECTNESS",
            "ORIGINALITY",
            "CLARITY",
            "IMPACT",
        ]
        aspect_label = {}  # aspect label for all papers

        mean = []
        aspect_label_count = {k: 0 for k in aspects}
        for id in self.datas.keys():
            data = self.datas[id]
            # print(f"Processing paper {data['paper_id']} with {len(data['reviews'])} reviews . . .")

            aspect_paper = {k: [] for k in aspects}  # aspect label for each paper
            for reviewer in data["reviews"]:
                for aspect in aspects:
                    if aspect in reviewer.keys():
                        aspect_paper[aspect].append(float(reviewer[aspect]))

                        if aspect == "RECOMMENDATION":
                            mean.append(float(reviewer[aspect]))
                            # find min and max aspect score
                            if float(reviewer[aspect]) < self.min_score:
                                self.min_score = float(reviewer[aspect])
                            if float(reviewer[aspect]) > self.max_score:
                                self.max_score = float(reviewer[aspect])
                    else:
                        pass

            # Compute aspect label for each paper
            if label_type == "avg":
                label_paper = {
                    k: np.mean(v) if len(v) > 0 else "not discussed"
                    for k, v in aspect_paper.items()
                }
                # aspect_paper = {k : np.nanmean(v) if ~np.isnan(np.nanmean(v)) else 'not discussed' for k, v in aspect_paper.items()}
            elif label_type == "most_freq":
                label_paper = {
                    k: max(set(v), key=v.count) if len(v) > 0 else "not discussed"
                    for k, v in aspect_paper.items()
                }
            else:
                raise ValueError(f"Invalid label type {label_type}")

            # count aspect label amount distribution
            for k, v in label_paper.items():
                if v != "not discussed":
                    aspect_label_count[k] += 1

            aspect_label[id] = label_paper

        print(f"total num of rated aspect : {aspect_label_count}")
        proportion_aspect_label_count = {
            k: round((v / len(aspect_label)), 2) for k, v in aspect_label_count.items()
        }
        print(f"proportion of rated aspect : {proportion_aspect_label_count}")
        print(f"total num of paper : {len(aspect_label)}")
        print(f"score range : [{self.min_score}, {self.max_score}]")

        return aspect_label

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        if self.datas[index]["paper_id"] not in self.skip_id_list:
            return (
                self.process_modifiable_contents(self.datas[index]),
                self.label[index],
            )
        else:
            return self.datas[index], self.label[index]

    def _normalize_string(self, s):
        return re.sub(r"\s+", " ", s).strip()
