"""
Recipes for DeepwordBug
"""

from textattack.constraints.overlap import LevenshteinEditDistance
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)
from textattack.transformations import (
    CompositeTransformation,
    WordSwapNeighboringCharacterSwap,
    WordSwapRandomCharacterDeletion,
    WordSwapRandomCharacterInsertion,
    WordSwapRandomCharacterSubstitution,
)

from attack.adv_reviewer.search_method.greedy_word_swap_wir import (
    GreedyWordSwapWIR,  # change output
)

from .attack_recipe import AttackRecipe


class DeepWordBug(AttackRecipe):
    """
    DeepWordBug: A Textual Adversarial Attack on Deep Learning Based NLP Models
    (https://arxiv.org/abs/1811.00158)
    """

    def __init__(self, config):
        super().__init__(config)

    def build(self):
        """
        Returns a DeepWordBug attack.
        """

        transformation = CompositeTransformation(
            [
                WordSwapNeighboringCharacterSwap(),
                WordSwapRandomCharacterSubstitution(),
                WordSwapRandomCharacterDeletion(),
                WordSwapRandomCharacterInsertion(),
            ]
        )

        # config
        constraints = [RepeatModification(), StopwordModification()]
        constraints.append(
            LevenshteinEditDistance(
                self.config["deepwordbug"]["levenshtein_edit_distance"]
            )
        )

        source = (
            "deepwordbug"
            if "query_budget" in self.config["deepwordbug"].keys()
            else "goal_function"
        )
        query_budget = self.config[source]["query_budget"]

        source = (
            "deepwordbug"
            if "top_k_wir" in self.config["deepwordbug"].keys()
            else "goal_function"
        )
        top_k_wir = self.config[source]["top_k_wir"]

        search_method = GreedyWordSwapWIR(
            query_budget=query_budget, top_k_wir=top_k_wir
        )

        return (transformation, constraints, search_method)
