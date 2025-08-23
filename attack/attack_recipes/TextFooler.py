"""
Recipes for TextFooler
"""

from textattack.constraints.grammaticality import PartOfSpeech

#  constraints
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder

# transformations
from textattack.transformations import WordSwapEmbedding

from attack.adv_reviewer.search_method.greedy_word_swap_wir import (
    GreedyWordSwapWIR,  # change output
)

from .attack_recipe import AttackRecipe


class TextFooler(AttackRecipe):
    """
    TextFooler: Is BERT Really Robust? A Strong Baseline for Natural Language Attack on Text Classification and Entailment.
    (https://arxiv.org/pdf/1907.11932v4)
    """

    def __init__(self, config):
        super().__init__(config)

    def build(self):
        """
        Returns a TextFooler attack.
        """
        transformation = WordSwapEmbedding(
            max_candidates=self.config["textfooler"]["max_candidates"]
        )
        stopwords = set(
            [
                "a",
                "about",
                "above",
                "across",
                "after",
                "afterwards",
                "again",
                "against",
                "ain",
                "all",
                "almost",
                "alone",
                "along",
                "already",
                "also",
                "although",
                "am",
                "among",
                "amongst",
                "an",
                "and",
                "another",
                "any",
                "anyhow",
                "anyone",
                "anything",
                "anyway",
                "anywhere",
                "are",
                "aren",
                "aren't",
                "around",
                "as",
                "at",
                "back",
                "been",
                "before",
                "beforehand",
                "behind",
                "being",
                "below",
                "beside",
                "besides",
                "between",
                "beyond",
                "both",
                "but",
                "by",
                "can",
                "cannot",
                "could",
                "couldn",
                "couldn't",
                "d",
                "didn",
                "didn't",
                "doesn",
                "doesn't",
                "don",
                "don't",
                "down",
                "due",
                "during",
                "either",
                "else",
                "elsewhere",
                "empty",
                "enough",
                "even",
                "ever",
                "everyone",
                "everything",
                "everywhere",
                "except",
                "first",
                "for",
                "former",
                "formerly",
                "from",
                "hadn",
                "hadn't",
                "hasn",
                "hasn't",
                "haven",
                "haven't",
                "he",
                "hence",
                "her",
                "here",
                "hereafter",
                "hereby",
                "herein",
                "hereupon",
                "hers",
                "herself",
                "him",
                "himself",
                "his",
                "how",
                "however",
                "hundred",
                "i",
                "if",
                "in",
                "indeed",
                "into",
                "is",
                "isn",
                "isn't",
                "it",
                "it's",
                "its",
                "itself",
                "just",
                "latter",
                "latterly",
                "least",
                "ll",
                "may",
                "me",
                "meanwhile",
                "mightn",
                "mightn't",
                "mine",
                "more",
                "moreover",
                "most",
                "mostly",
                "must",
                "mustn",
                "mustn't",
                "my",
                "myself",
                "namely",
                "needn",
                "needn't",
                "neither",
                "never",
                "nevertheless",
                "next",
                "no",
                "nobody",
                "none",
                "noone",
                "nor",
                "not",
                "nothing",
                "now",
                "nowhere",
                "o",
                "of",
                "off",
                "on",
                "once",
                "one",
                "only",
                "onto",
                "or",
                "other",
                "others",
                "otherwise",
                "our",
                "ours",
                "ourselves",
                "out",
                "over",
                "per",
                "please",
                "s",
                "same",
                "shan",
                "shan't",
                "she",
                "she's",
                "should've",
                "shouldn",
                "shouldn't",
                "somehow",
                "something",
                "sometime",
                "somewhere",
                "such",
                "t",
                "than",
                "that",
                "that'll",
                "the",
                "their",
                "theirs",
                "them",
                "themselves",
                "then",
                "thence",
                "there",
                "thereafter",
                "thereby",
                "therefore",
                "therein",
                "thereupon",
                "these",
                "they",
                "this",
                "those",
                "through",
                "throughout",
                "thru",
                "thus",
                "to",
                "too",
                "toward",
                "towards",
                "under",
                "unless",
                "until",
                "up",
                "upon",
                "used",
                "ve",
                "was",
                "wasn",
                "wasn't",
                "we",
                "were",
                "weren",
                "weren't",
                "what",
                "whatever",
                "when",
                "whence",
                "whenever",
                "where",
                "whereafter",
                "whereas",
                "whereby",
                "wherein",
                "whereupon",
                "wherever",
                "whether",
                "which",
                "while",
                "whither",
                "who",
                "whoever",
                "whole",
                "whom",
                "whose",
                "why",
                "with",
                "within",
                "without",
                "won",
                "won't",
                "would",
                "wouldn",
                "wouldn't",
                "y",
                "yet",
                "you",
                "you'd",
                "you'll",
                "you're",
                "you've",
                "your",
                "yours",
                "yourself",
                "yourselves",
            ]
        )
        constraints = [RepeatModification(), StopwordModification(stopwords=stopwords)]

        constraints.append(
            WordEmbeddingDistance(
                min_cos_sim=self.config["textfooler"]["min_word_cos_sim"]
            )
        )
        constraints.append(PartOfSpeech(allow_verb_noun_swap=True))
        use_constraint = UniversalSentenceEncoder(
            threshold=self.config["textfooler"]["min_sentence_cos_sim"],
            metric="angular",
            compare_against_original=False,
            window_size=15,
            skip_text_shorter_than_window=True,
        )
        constraints.append(use_constraint)

        source = (
            "textfooler"
            if "query_budget" in self.config["textfooler"].keys()
            else "goal_function"
        )
        query_budget = self.config[source]["query_budget"]

        source = (
            "textfooler"
            if "top_k_wir" in self.config["textfooler"].keys()
            else "goal_function"
        )
        top_k_wir = self.config[source]["top_k_wir"]

        search_method = GreedyWordSwapWIR(
            wir_method="delete", query_budget=query_budget, top_k_wir=top_k_wir
        )

        return (transformation, constraints, search_method)
