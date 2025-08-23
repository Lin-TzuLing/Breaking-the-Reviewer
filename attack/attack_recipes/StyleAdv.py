"""
Recipes for StyleAdv
"""

from attack.adv_reviewer.search_method.brute_force_search import BruteForceSearch
from attack.adv_reviewer.transformations.style_transformation import StyleTransformation

from .attack_recipe import AttackRecipe


class StyleAdv(AttackRecipe):
    """
    DeepWordBug: A Textual Adversarial Attack on Deep Learning Based NLP Models
    (https://arxiv.org/abs/1811.00158)
    """

    def __init__(self, config):
        super().__init__(config)

    def build(self):
        """
        Returns a StyleAdv attack.
        """

        # self-defined transformations
        transformation = StyleTransformation(self.config)

        constraints = []

        search_method = BruteForceSearch()

        return (transformation, constraints, search_method)
