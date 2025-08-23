import glob
import json
import os
import re
import warnings
from collections import Counter
from typing import Optional

import numpy as np
from torch.utils.data import Dataset

from data.utils.lcs_match import add_modifiable_markers
from prompt import PromptTemplate

warnings.filterwarnings("ignore")


def collate_fn_agentreview(input_metainfo, input_contents):

    def collate_fn(batch):
        x, y = [], []
        for _, (data, label) in enumerate(batch):
            x.append(data)
            y.append(label)
        return (x, y)

    return collate_fn


class AgentReviewDataset(Dataset):
    """
    AgentReview dataset.

    Args:
        dataset_name (:obj:`str`, optional): the name of the dataset. Default to 'iclr_2017'.
        dataset_dir (:obj:`str`): the path to the raw dataset. Default to 'data/raw_dataset/PeerRead/data/iclr_2017'.
        dataset_mode (:obj:`str`): the split of data. choices = ['all', 'train', 'dev', 'test']. Default to 'all'.
        data_label_type (:obj:`str`): the score label type. choices = ['avg', 'most_feq']. Default to 'avg'.
    """

    def __init__(
        self,
        dataset_name: Optional[str] = "",
        dataset_dir: str = "data/dataset/AgentReview",
        dataset_mode: Optional[str] = "",
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

        # ====================================Dataset path==========================================
        self.DIR_DATASET_BASE = dataset_dir
        PARSED_PDFS_RE = "/*/parsed_pdfs/*/*.json"
        NOTES_RE = "/*/notes/*/*.json"

        parsed_pdf_path_list, notes_path_list = self.get_papers_and_notes(
            dataset_base_path=self.DIR_DATASET_BASE,
            re_parsed_pdfs=PARSED_PDFS_RE,
            re_notes=NOTES_RE,
        )

        self.parsed_pdf_path_list = [
            p.replace(self.DIR_DATASET_BASE, "") for p in parsed_pdf_path_list
        ]
        parsed_pdf_path_list.sort()
        self.notes_path_list = [
            n.replace(self.DIR_DATASET_BASE, "") for n in notes_path_list
        ]
        notes_path_list.sort()
        print(f"num of papers : {len(parsed_pdf_path_list)}")
        # ==========================================================================================

        print("=======================================================")
        print(f"Loading {dataset_name} dataset . . .")

        self.datas = self.read_data(parsed_pdf_path_list, notes_path_list)

        self.min_score, self.max_score = float("inf"), float("-inf")
        self.label = self.compute_labels(data_label_type)

        # process read manual review for long common subsequence match
        self.manual_review, self.lcs_split_level = self.read_manual_review(
            self.manual_review_root
        )

        assert len(self.label) == len(self.datas)
        print(f"\n{dataset_name} dataset loaded successfully!")
        print("=======================================================")

    def read_data(self, parsed_pdf_path_list, notes_path_list):
        """
        Read data from the raw dataset.

        Args:
            parsed_pdf_path_list (:obj:`list`): the list of parsed pdf path.
            notes_path_list (:obj:`list`): the list of notes path.

        Returns:
            datas (:obj:`dict`): the data of all papers.
        """

        # read review files
        data_review = {}
        num_avg_reviews, num_min_reviews = 0, 100
        review_key_mapping = {
            "2020": "review",
            "2021": "review",
            "2022": "main_review",
            "2023": "summary_of_the_review",
        }
        rating_key_mapping = {
            "2020": "rating",
            "2021": "rating",
            "2022": "recommendation",
            "2023": "recommendation",
        }

        print(f"Reading review files from '{self.path}/notes' . . .")
        for filename in notes_path_list:
            paper_id = self.rename(filename).split(".")[0]
            year = paper_id.split("_")[0][4:]

            with open(filename, "r", encoding="utf-8") as f:
                data = json.loads(f.read())

            # filter out empty comments and make sure replies are from reviewers not authors
            comments = [
                (
                    r["content"][review_key_mapping[year]],
                    r["content"][rating_key_mapping[year]].split(":")[0],
                )  # (review, rating)
                for r in data["details"]["replies"]
                if any(key in r["content"] for key in rating_key_mapping.values())
                and r["content"][review_key_mapping[year]] != ""
            ]

            data["reviews"] = sorted(list(set(comments)))
            data = {
                k: v for k, v in data.items() if k in ["number", "reviews", "content"]
            }
            data_review[paper_id] = data

            if len(data_review[paper_id]["reviews"]) < num_min_reviews:
                num_min_reviews = len(data_review[paper_id]["reviews"])
            num_avg_reviews += len(data_review[paper_id]["reviews"])

        print(f"num of reviews : {len(data_review)}")
        print(f"avg reviews per paper: {(num_avg_reviews/len(data_review)):.4f}")
        print(f"min reviews per paper: {num_min_reviews}")

        # read paper files
        data_paper = {}
        print(f"Reading paper files from '{self.path}/parsed_pdfs' . . .")
        for filename in parsed_pdf_path_list:
            paper_id = self.rename(filename).split(".")[0]
            with open(filename, "r", encoding="utf-8") as f:
                data_paper[paper_id] = json.loads(f.read())
        print(f"num of papers : {len(data_paper)}")

        assert data_review.keys() == data_paper.keys()

        # merge data
        datas = {}
        num_abstract = 0
        for id in data_paper.keys():
            review = data_review[id]
            paper = data_paper[id]

            assert (
                id.split("_")[-1] == str(review["number"])
                and id.split("_")[-1] == paper["name"].split(".")[0]
            )

            # abstract and sections
            paper_contents = {}
            paper_contents["Abstract"] = review["content"]["abstract"]
            num_abstract += len(paper_contents["Abstract"].split())
            if paper["metadata"]:
                for sec in paper["metadata"]["section"]:
                    paper_contents[f"Section {sec['heading']}"] = sec["text"]

            paper["paper_contents"] = paper_contents
            paper.pop("metadata")
            paper.pop("name")
            review.pop("content")
            review.pop("number")

            datas[self.re_index] = {
                **review,
                **paper,
                **{"paper_id": id},
            }  # merge two dict
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
            files = os.listdir(manual_review_root)
            for filename in files:
                path = os.path.join(manual_review_root, filename)
                paper_id = os.path.splitext(os.path.basename(path))[0]

                with open(path, "r") as f:
                    content = f.read()

                result = PromptTemplate.parseManualReview(content)
                if len(result) != 2 or len(result[0]) == 0 or len(result[1]) != 8:
                    print(f"paper {paper_id} has no manual review")
                else:
                    manual_review[paper_id] = result

            print(f"num of manual review : {len(manual_review)}")

            char_level_attack = ["DeepWordBug", "PuncAttack"]
            word_level_attack = ["TextFooler", "BertAttack", "Checklist"]
            sent_level_attack = ["StyleAdv", "SyntacticAdv"]
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
            modifiable_contents_dict (:obj:`dict`): the modifiable contents of the paper.
        """

        # split paper content and meta information
        paper_id = datas["paper_id"]
        data_content = datas["paper_contents"]
        data_meta = {k: v for k, v in datas.items() if k != "paper_contents"}

        modifiable_contents_dict = {}
        # add meta information, e.g. paper_id
        if self.input_metainfo:
            modifiable_contents_dict.update(
                {k: v for k, v in data_meta.items() if k in self.input_metainfo}
            )
        else:
            modifiable_contents_dict.update(data_meta)

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
                modifiable_contents_dict["paper_content"] = content
            else:
                modifiable_contents_dict["paper_content"] = (
                    content_str  # no manual review, no unmodifiable markers
                )
                print(f"paper {paper_id} not in manual review")

        print(f"paper {paper_id} processed")
        return modifiable_contents_dict

    def compute_labels(self, label_type):
        """
        Process aspect score label for each paper & example output template.

        Args:
            label_type (:obj:`str`): the score label type. choices = ['avg', 'most_feq'].

        Returns:
            paper_label (:obj:`dict`): aspect label for all papers.
        """

        paper_label = {}

        for id in self.datas.keys():

            data = self.datas[id]
            # print(f"Processing paper {data['paper_id']} with {len(data['reviews'])} reviews . . .")

            paper_score = []
            for review in data["reviews"]:

                score = review[1]
                paper_score.append(float(score))

                # find min and max score
                if float(score) < self.min_score:
                    self.min_score = float(score)
                if float(score) > self.max_score:
                    self.max_score = float(score)

            # Compute aspect label for each paper
            if label_type == "avg":
                paper_score = np.mean(np.array(paper_score))
            elif label_type == "most_freq":
                counter = Counter(paper_score)
                paper_score, _ = counter.most_common(1)[0]
            else:
                raise ValueError(f"Invalid label type {label_type}")

            paper_label[id] = paper_score

        print(f"total num of paper : {len(paper_label)}")
        print(f"score range : [{self.min_score}, {self.max_score}]")

        return paper_label

    # get two sources
    def get_papers_and_notes(
        self, dataset_base_path: str, re_parsed_pdfs: str, re_notes: str
    ):
        parsed_pdf_path_list = glob.glob(f"{dataset_base_path}{re_parsed_pdfs}")
        notes_path_list = glob.glob(f"{dataset_base_path}{re_notes}")
        return parsed_pdf_path_list, notes_path_list

    # dictionary key
    def rename(self, target_path: str) -> str:
        """Rename the output to avoid the name confliction."""
        dir_path, fname = os.path.split(target_path)
        dir_path = dir_path.split(f"{self.DIR_DATASET_BASE}/")[-1]
        pre_info = dir_path.replace("/parsed_pdfs/", "_").replace("/notes/", "_")
        return f"{pre_info}_{fname}"

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
