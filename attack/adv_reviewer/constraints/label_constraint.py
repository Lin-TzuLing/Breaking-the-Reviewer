# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from textattack.constraints import PreTransformationConstraint


class LabelConstraint(PreTransformationConstraint):
    """
    A constraint that does not allow to attack the labels (or any words that is important for tasks) in the prompt.
    """

    def __init__(self, labels=[]):
        self.labels = [label.lower() for label in labels]
        self.verbose_modifiable = True

    def _get_modifiable_indices(self, current_text):
        modifiable_indices = set()
        modifiable_words = []

        unmodifiable_section_flag = False
        for i, word in enumerate(current_text.words):
            if word == "UnmodifiableStart":
                unmodifiable_section_flag = True
            if str(word).lower() not in self.labels and not unmodifiable_section_flag:
                modifiable_words.append(word)
                modifiable_indices.add(i)
            if word == "UnmodifiableEnd":
                unmodifiable_section_flag = False
        return modifiable_indices

    def check_compatibility(self, transformation):
        """
        It is always true.
        """
        return True
