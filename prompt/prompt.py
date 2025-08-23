import re


class PromptTemplate:
    """
    A template class for generating prompts for automated research paper review tasks.
    This class creates structured prompts for AI models to generate ICLR-style reviews
    with tagged sentences and numerical scores across multiple evaluation aspects.
        aspect_tag_types (list, optional): List of aspect tag types for categorizing
            review sentences. Defaults to predefined tags including SUMMARY,
            MOTIVATION, SUBSTANCE, ORIGINALITY, SOUNDNESS, CLARITY, REPLICABILITY,
            and MEANINGFUL COMPARISON with POSITIVE/NEGATIVE variants.
        aspect_score_types (list, optional): List of scoring aspects for numerical
            evaluation. Defaults to OVERALL, SUBSTANCE, APPROPRIATENESS,
            MEANINGFUL_COMPARISON, SOUNDNESS_CORRECTNESS, ORIGINALITY, CLARITY, IMPACT.
        explain (bool, optional): Whether to include score explanations in the output
            format. Defaults to False.
        **kwargs: Additional keyword arguments.
    Attributes:
        explain (bool): Flag indicating whether explanations are required.
        template (dict): Dictionary containing input and output templates.
    Methods:
        generate(heading, content): Generate a single prompt from paper section.
        generate_batch(batch_data): Generate multiple prompts from batch data.
        removed_unmodifiable_tokens(text): Remove special unmodifiable tokens.
        parseOutputText(texts, explain): Parse AI-generated review outputs.
        parseManualReview(text): Parse manually written review text.
    Example:
        >>> prompt_gen = PromptTemplate(explain=True)
        >>> prompt = prompt_gen.generate("Abstract", "This paper presents...")
        >>> parsed = PromptTemplate.parseOutputText([ai_output], explain=True)
    """

    def __init__(
        self,
        aspect_tag_types=[
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
        aspect_score_types=[
            "OVERALL",
            "SUBSTANCE",
            "APPROPRIATENESS",
            "MEANINGFUL_COMPARISON",
            "SOUNDNESS_CORRECTNESS",
            "ORIGINALITY",
            "CLARITY",
            "IMPACT",
        ],
        explain=False,
        **kwargs,
    ):

        self.explain = explain
        template = {}

        # prepare the input template
        if self.explain:
            input = (
                "You are a professional reviewer in computer science and machine learning. "
                "Based on the given content of a research paper, you need to write a review in ICLR style "
                "  "
            )
            input += (
                "tags types: "
                + ", ".join([f"[{tag}]" for tag in aspect_tag_types])
                + ". "
            )
            input += (
                "Your total output should not surpass 500 tokens, make sure to include both positive and negative aspects. "
                # explanation
                "Also, you need to predict the review score for several aspects based on the generated review, providing an explanation of each aspect in less than 30 tokens. "
                "Choose a integer score from 1 to 10, higher score means better paper quality. "
            )
        else:
            input = (
                "You are a professional reviewer in computer science and machine learning. "
                "Based on the given content of a research paper, you need to write a review in ICLR style "
                "and tag sentences with the corresponding tag type at the beginning of sequence: "
            )
            input += (
                "tags types: "
                + ", ".join([f"[{tag}]" for tag in aspect_tag_types])
                + ". "
            )
            input += (
                "Your total output should not surpass 500 tokens, make sure to include both positive and negative aspects. "
                "Also, you need to predict the review score in several aspects without explanation. "
                "Choose a integer score from 1 to 10, higher score means better paper quality. "
            )
        template["input"] = input

        # prepare the output template
        example_output = "Please strictly follow the format of Example output: "
        example_output += "1. REVIEW: tagged sequences. "
        example_output += (
            "2. REVIEW SCORE: "
            + ", ".join([f"{label}: score" for label in aspect_score_types])
            + "."
        )
        if self.explain:
            example_output += (
                "3. REVIEW SCORE EXPLANATION: "
                + ", ".join([f"{label}: explanation" for label in aspect_score_types])
                + "."
            )
        template["example_output"] = example_output

        self.template = template

    def generate(self, heading, content):
        """
        Generate the prompt for the task.

        Args:
            heading (:obj:`str`): Heading of the paper section.
            content (:obj:`str`): Content of the paper.

        Returns:
            str: The generated prompt.
        """

        prompt_input = self.template["input"] + "\n"
        prompt_input += heading + ": " + content + "\n"
        prompt_input += self.template["example_output"]

        return prompt_input

    def generate_batch(self, batch_data):
        """
        Generate the prompt for the task.

        Args:
        batch_data (:obj:`list`): List of tuples (heading, content) of the paper section.
        
        Return:
            list: List of generated prompts.
        """

        batch_prompt = []

        for data in batch_data:

            data = self.removed_unmodifiable_tokens(data)

            prompt_input = self.template["input"] + "\n"
            prompt_input += data + "\n"
            prompt_input += self.template["example_output"]
            batch_prompt.append(prompt_input)

        return batch_prompt

    def removed_unmodifiable_tokens(self, text):
        """
        Remove unmodifiable tokens from the text.
        """

        return text.replace("<UnmodifiableStart>", "").replace("<UnmodifiableEnd>", "")

    @staticmethod
    def parseOutputText(texts: list, explain: bool = False):
        """
        Parse AI-generated review outputs.

        Args:
            texts (list): List of AI-generated review output texts.
            explain (bool): Whether to include explanations in the output.

        Returns:
            list: List of parsed review outputs.
        """
        parsed_content = []
        for text in texts:
            if explain:
                # Extract REVIEW content
                review_match = re.search(r"1\. REVIEW:\s*(.*?)\n\n", text, re.DOTALL)
                # review_match = re.search(r".*?REVIEW:\s*(.*?)\n\n", text, re.DOTALL)
                review_text = review_match.group(1).strip() if review_match else ""

                # Extract REVIEW SCORE content
                score_match = re.search(
                    r"2\. REVIEW SCORE:\s*(.*?)\n\n", text, re.DOTALL
                )
                # score_match = re.search(r".*?REVIEW SCORE:\s*(.*?)\n\n", text, re.DOTALL)
                score_text = score_match.group(1).strip() if score_match else ""

                # Extract REVIEW SCORE EXPLANATION content
                explanation_match = re.search(
                    r"3\. REVIEW SCORE EXPLANATION:\s*(.*)", text, re.DOTALL
                )
                # explanation_match = re.search(r".*?REVIEW SCORE EXPLANATION:\s*(.*)", text, re.DOTALL)
                explanation_text = (
                    explanation_match.group(1).strip() if explanation_match else ""
                )

                # Create dictionaries
                review_dict = {
                    key.replace("*", ""): value
                    for key, value in re.findall(
                        r"\[(.*?)\]\s*(.*?)(?=\s*\[|\Z)", review_text
                    )
                }
                score_dict = {
                    key.replace("*", ""): int(value)
                    for key, value in re.findall(r"([\w\s]+):\s*(\d+)", score_text)
                }
                explanation_dict = {
                    key.replace("*", ""): value
                    for key, value in re.findall(
                        r"([\w\s]+):\s*(.*?)(?=\s*\w[\w\s]*:|$)", explanation_text
                    )
                }

                # Append dictionaries to lists
                parsed_content.append([review_dict, score_dict, explanation_dict])
            else:
                # Extract REVIEW content
                # review_match = re.search(r"1\. REVIEW:\s*(.*?)2\. REVIEW SCORE:", text, re.DOTALL)
                review_match = re.search(
                    r"(?:\d+\.\s*|###\s*)REVIEW:\s*(.*?)(?=\s*(?:\d+\.\s*|###\s*)REVIEW SCORE:|\Z)",
                    text,
                    re.DOTALL,
                )
                review_text = review_match.group(1).strip() if review_match else ""

                # Extract REVIEW SCORE content
                # score_match = re.search(r"2\. REVIEW SCORE:\s*(.*)", text, re.DOTALL)
                score_match = re.search(
                    r"(?:\d+\.\s*|###\s*)REVIEW SCORE:\s*(.*)", text, re.DOTALL
                )
                score_text = score_match.group(1).strip() if score_match else ""

                # Create dictionaries
                review_dict = {
                    key.replace("*", ""): value
                    for key, value in re.findall(
                        r"\[(.*?)\]\s*(.*?)(?=\s*\[|\Z)", review_text
                    )
                }
                score_dict = {
                    key.replace("*", ""): int(value)
                    for key, value in re.findall(r"(\**\w+\**):\s*(\d+)", score_text)
                }

                # Append dictionaries to lists
                parsed_content.append([review_dict, score_dict])

        return parsed_content

    @staticmethod
    def parseManualReview(text: str):
        """
        Parse manually written review text.

        Args:
            text (str): The manually written review text.

        Returns:
            list: A list containing the parsed review and score dictionaries.
        """

        # 1. Unescape "\[...]" to "[...]"
        text = (
            text.replace(r"\[", "[")
            .replace(r"\]", "]")
            .replace("*", "")
            .replace("**", "")
        )

        # 2. Match REVIEW section (supports "1. **REVIEW:**", etc.)
        review_match = re.search(
            r"(?:\d+\.\s*)?\*?\*?REVIEW\*?\*?:?\s*\n?(.*?)(?:\n\s*)?(?:\d+\.\s*)?\*?\*?REVIEW SCORE\*?\*?:",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        review_text = review_match.group(1).strip() if review_match else ""

        # 3. Match REVIEW SCORE section
        score_match = re.search(
            r"REVIEW SCORE\s*:?[\s\n]*(.*)", text, re.DOTALL | re.IGNORECASE
        )
        score_text = score_match.group(1).strip() if score_match else ""

        # 4. Extract tagged key-value blocks (multiline-safe)
        review_dict = {
            key.strip(): value.strip()
            for key, value in re.findall(
                r"\[([A-Z_ ]+)\](.*?)(?=\n\[|$)", review_text, re.DOTALL
            )
        }

        # 5. Extract score key-values (with optional markdown-style *)
        score_dict = {
            key.strip(): int(value.strip())
            for key, value in re.findall(r"\*?\s*([A-Za-z_]+)\s*:\s*(\d+)", score_text)
        }

        parsed_content = [review_dict, score_dict]

        return parsed_content
