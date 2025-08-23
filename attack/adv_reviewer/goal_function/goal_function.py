# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# Source Attribution:
# The majority of this code is derived from the following sources:
# - TextAttack GitHub Repository: https://github.com/QData/TextAttack
# - Reference Paper: Morris, John X., et al. "TextAttack: A Framework for Adversarial Attacks, Data Augmentation, and Adversarial Training in NLP." arXiv preprint arXiv:2005.05909 (2020).

import asyncio
import datetime
from abc import ABC, abstractmethod

from textattack.goal_function_results import TextToTextGoalFunctionResult
from textattack.goal_function_results.goal_function_result import (
    GoalFunctionResultStatus,
)
from textattack.shared.utils import ReprMixin


class GoalFunction(ReprMixin, ABC):
    """Evaluates how well a perturbed attacked_text object is achieving a
    specified goal.

    Args:
        model (:class:`LLMModelWrapper.OpenAIModelWrapper.GPT4oMini`):
            The victim model to attack.
        maximizable(:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether the goal function is maximizable, as opposed to a boolean result of success or failure.
            if False, the goal function is a boolean result of success or failure.(check is_goal_complete)
        query_budget (:obj:`float`, `optional`, defaults to :obj:`float("in")`):
            The maximum number of model queries allowed.
    """

    def __init__(
        self,
        model,
        prompter,
        maximizable=False,
        query_budget=float("inf"),
        score_threshold=0.0,
        **kwargs,
    ):
        self.model = model
        self.prompter = prompter
        self.maximizable = maximizable
        self.query_budget = query_budget
        self.score_threshold = score_threshold

    def init_attack_example(self, attacked_text):
        """Called before attacking ``attacked_text`` to 'reset' the goal
        function and set properties for this example."""
        self.initial_attacked_text = attacked_text
        self.ground_truth_output = None
        self.num_queries = 0
        result, _ = self.get_results(attacked_text, check_skip=True)
        return result

    @abstractmethod
    def get_results(self, attacked_text_list, check_skip=False):
        """For each attacked_text object in attacked_text_list, returns a result consisting of
        whether or not the goal has been achieved, the output for display purposes, and a score.

        Additionally returns whether the search is over due to the query budget.
        """
        raise NotImplementedError()

    def _get_goal_status(
        self, model_output, gt_output, attacked_text, check_skip=False
    ):
        should_skip = check_skip and self._should_skip(
            model_output, gt_output, attacked_text
        )
        if should_skip:
            return GoalFunctionResultStatus.SKIPPED
        if self.maximizable:
            return GoalFunctionResultStatus.MAXIMIZING
        if self._is_goal_complete(model_output, gt_output, attacked_text):
            return GoalFunctionResultStatus.SUCCEEDED
        return GoalFunctionResultStatus.SEARCHING

    @abstractmethod
    def _is_goal_complete(self, model_output, gt_output, attacked_text):
        raise NotImplementedError()

    def _should_skip(self, model_output, gt_output, attacked_text):
        return self._is_goal_complete(model_output, gt_output, attacked_text)

    @abstractmethod
    def _get_score(self, model_output, gt_output, attacked_text):
        raise NotImplementedError()

    @abstractmethod
    def _call_model(self, attacked_text_list):
        """Calls the model on a list of attacked_text objects."""
        raise NotImplementedError()

    @abstractmethod
    def _goal_function_result_type(self):
        """Returns the class of this goal function's results."""
        raise NotImplementedError()


class AdvReviewGoalFunction(GoalFunction):
    """A goal function defined on a model that outputs a probability for some
    number of classes."""

    def __init__(
        self, model, prompter, query_budget, score_threshold, verbose=False, **kwargs
    ):
        super().__init__(
            model=model,
            prompter=prompter,
            query_budget=query_budget,
            score_threshold=score_threshold,
            **kwargs,
        )
        self.output_explanation = prompter.explain

    def _call_model(self, attacked_text_list):
        print(f"Num of Attack Samples: {len(attacked_text_list)}")
        attacked_text = [attacked_text.text for attacked_text in attacked_text_list]

        prompt = self.prompter.generate_batch(attacked_text)

        # result
        print(f"{datetime.datetime.now()} prompting start ... ")

        output = asyncio.run(self.model(prompt))

        print(f"{datetime.datetime.now()} prompting done ... ")

        review_content = self.valid_output(output, prompt)

        return review_content

    def _is_goal_complete(self, model_output, gt_output, _):
        """
        Args:
            model_output: [review_aspect, review_score], take attacked paper content as input.
            gt_output: [review_aspect, review_score], take clean paper content as input.
        return:
            whether the goal is achieved. (bool)
        """

        # set model output of clean paper as ground truth.
        score_increase = float(
            self._get_score(model_output) - sum(gt_output[1].values())
        )
        return score_increase > self.score_threshold

    def _get_score(self, model_output):
        """
        Args:
            model_output: [review_aspect, review_score]
        return:
            sum of review_score
        """
        return sum(model_output[1].values())

    def get_results(self, attacked_text_list, check_skip=False):
        results = []

        if self.query_budget < float("inf"):
            queries_left = self.query_budget - self.num_queries
            attacked_text_list = attacked_text_list[:queries_left]
        self.num_queries += len(attacked_text_list)

        model_outputs = self._call_model(attacked_text_list)

        if self.ground_truth_output is None:
            self.ground_truth_output = list(zip(*model_outputs))[
                0
            ]  # set model output of clean paper as ground truth.

        for attacked_text, raw_output in zip(attacked_text_list, zip(*model_outputs)):
            goal_status = self._get_goal_status(
                raw_output,
                self.ground_truth_output,
                attacked_text,
                check_skip=check_skip,
            )
            goal_function_score = self._get_score(raw_output)

            results.append(
                self._goal_function_result_type()(
                    attacked_text=attacked_text,
                    raw_output=raw_output,
                    output=raw_output,
                    goal_status=goal_status,
                    score=goal_function_score,
                    num_queries=self.num_queries,
                    ground_truth_output=self.ground_truth_output,
                )
            )

        return results, self.num_queries == self.query_budget

    def _goal_function_result_type(self):
        """Returns the class of this goal function's results."""
        return TextToTextGoalFunctionResult

    def valid_output(self, output, prompt):
        def parse_output(output):
            parsed = list(
                zip(*self.prompter.parseOutputText(output, self.output_explanation))
            )
            return list(map(list, parsed))

        parsed_reviews = parse_output(output)

        try:
            count = 3 if len(parsed_reviews[0]) > 1 else float("inf")
        except:
            print(
                f"Error occurred while parsing output:\nOutput: {output}\nParsed: {parsed_reviews}"
            )
        while True and count > 0:
            count -= 1

            # Step 1: Find all invalid indices (parsing failed) which require regeneration
            invalid_indices = [
                idx
                for idx in range(len(parsed_reviews[0]))
                if len(parsed_reviews[1][idx]) != 8 or len(parsed_reviews[0][idx]) <= 1
            ]

            # If there are no invalid indices, break the loop
            if not invalid_indices:
                break
            print(f"Re-generating output for prompts at indices: {invalid_indices}")

            # Step 2: Extract all prompts that need regeneration
            invalid_prompts = [prompt[idx] for idx in invalid_indices]

            # Step 3: Regenerate outputs for all invalid prompts
            new_outputs = asyncio.run(self.model(invalid_prompts))
            new_parsed = parse_output(new_outputs)

            # Step 4: Replace the corresponding positions in the original `parsed_reviews`
            for i in range(len(parsed_reviews)):
                for pos, idx in enumerate(invalid_indices):
                    parsed_reviews[i][idx] = new_parsed[i][pos]

        return parsed_reviews if self.output_explanation else parsed_reviews[:2]
