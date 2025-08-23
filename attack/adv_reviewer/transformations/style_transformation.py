"""
StyleTransformation Class
-------------------------------
style transformations by
Mind the Style of Text! Adversarial and Backdoor Attacks Based on Text Style Transfer (Qi et al., 2021)
"""

import os
import re

import nltk
from nltk.tokenize import sent_tokenize
from textattack.shared import AttackedText
from textattack.transformations.sentence_transformations import SentenceTransformation

from .utils.style.inference_utils import GPT2Generator

nltk.download("punkt")


class StyleTransformation(SentenceTransformation):
    """A base class for style transformation."""

    def __init__(self, config):
        self.style = config["styleadv"]["style"]
        # self.max_candidates = config["styleadv"]["max_candidates"]

        # define the paraphraser
        self.paraphraser_path = os.path.join(
            config["styleadv"]["paraphraser_path"], self.style
        )
        self.paraphraser = GPT2Generator(self.paraphraser_path, upper_length="same_5")
        self.paraphraser.modify_p(top_p=0.6)

    def __call__(
        self,
        current_text,
        pre_transformation_constraints=[],
        indices_to_modify=None,
        shifted_idxs=False,
        return_indices=False,
    ):
        """
        Returns a list of all possible transformations for ``current_text``.
        Applies the ``pre_transformation_constraints`` then calls ``_get_transformations``.

        Args:
            current_text: The ``AttackedText`` to transform.
            pre_transformation_constraints: The ``PreTransformationConstraint`` to apply before
                beginning the transformation.
            indices_to_modify: Which word indices should be modified as dictated by the
                ``SearchMethod``.
            shifted_idxs (bool): Whether indices could have been shifted from
                their original position in the text.
            return_indices (bool): Whether the function returns indices_to_modify
                instead of the transformed_texts.
        """
        if indices_to_modify is None:
            indices_to_modify = set(range(len(current_text.words)))
            # If we are modifying all indices, we don't care if some of the indices might have been shifted.
            shifted_idxs = False
        else:
            indices_to_modify = set(indices_to_modify)

        if shifted_idxs:
            indices_to_modify = set(
                current_text.convert_from_original_idxs(indices_to_modify)
            )

        for constraint in pre_transformation_constraints:
            indices_to_modify = indices_to_modify & constraint(current_text, self)

        if return_indices:
            return indices_to_modify

        transformed_texts = self._get_transformations(current_text, indices_to_modify)
        for text in transformed_texts:
            text.attack_attrs["last_transformation"] = self
        return transformed_texts

    def transform_batch(self, text_li: list):

        # sent_idx = [e[0] for e in text_li]
        # sent_text = [e[1] for e in text_li]
        generate_text, _ = self.paraphraser.generate_batch(text_li)

        # make sure the generated text is different from the original text and not empty
        iter = 0
        for i in range(len(generate_text)):
            while (
                generate_text[i].isspace() or generate_text[i] == text_li[i]
            ) and iter <= 100:
                generate_text[i] = self.paraphraser.generate(text_li[i])
                iter += 1

        # generations = [(text_li[i], generate_text[i]) for i in range(len(generate_text))]
        return generate_text

    def _get_transformations(self, current_text, indices_to_modify):
        text = current_text.text
        # 正則表達式匹配每個 <UnmodifiableStart> 和 <UnmodifiableEnd> 之間的段落
        pattern = r"(<UnmodifiableEnd>.*?<UnmodifiableStart>)"
        matches = re.findall(pattern, text, flags=re.DOTALL)

        # 處理每個匹配的段落
        outputs = []
        # print(f"num of matches: {len(matches)}")

        for i, match in enumerate(matches):
            # 提取每個段落中的 <UnmodifiableStart> 和 <UnmodifiableEnd> 之間的內容
            content = re.sub(r"<Unmodifiable(End|Start)>", "", match)
            content = content.strip()

            # 將段落分成句子
            sentences = sent_tokenize(content)
            # print(sentences)
            # print('------------------------')
            # 假設 transform_batch 返回的是對應句子轉換後的列表
            transformed_sentences = self.transform_batch(sentences)
            # print(transformed_sentences)
            # print('------------------------')

            # 每次替換一個句子，這裡可以是替換第i個句子
            # print(f"match: {i}, num of sentences: {len(sentences)}")
            for idx in range(len(sentences)):
                sent_combine = " ".join(
                    sentences[:idx]
                    + [transformed_sentences[idx]]
                    + sentences[idx + 1 :]
                )
                output = f"<UnmodifiableEnd>{sent_combine}<UnmodifiableStart>"

                # 將處理過的段落插回原始文本，使用正則表達式來替換原來的段落
                final_text = text
                final_text = final_text.replace(match, output)
                outputs.append(final_text)

        # print(len(outputs))
        transformed_result = [AttackedText(text) for text in outputs]
        # for text in transformed_result:
        #     print(f"Transformed text: {text.text[:1000]}")
        #     print('------------------------')
        # break
        #
        # print(len(transformed_result))
        # print(transformed_result[0].text[:1000])
        # print('------------------------')

        return transformed_result

    # def strip_punctuation(self, word):
    #     return re.sub(r'[^\w\s]', '', word)

    # def find_ranges(self, index_list): # find the ranges of the index list containing continuous numbers
    #     index_list.sort()
    #     ranges = []
    #     start = index_list[0]

    #     for i in range(1, len(index_list)):
    #         if index_list[i] != index_list[i - 1] + 1:
    #             ranges.append((start, index_list[i - 1]))
    #             start = index_list[i]

    #     ranges.append((start, index_list[-1]))
    #     return ranges

    # def split_text(self, text):
    #     pattern = r"(<UnmodifiableStart>|<UnmodifiableEnd>|[^<\s]+)"
    #     parts = re.findall(pattern, text)
    #     return parts

    # def _get_transformations(self, current_text, indices_to_modify):

    #     print(current_text)
    #     exit()
    #     # print(current_text.words)
    #     # exit()
    #     # print(f"Original text: {current_text.text}")
    #     # print(f"Indices to modify: {indices_to_modify}")
    #     # print(f"Words to modify: {[current_text.words[idx] for idx in indices_to_modify]}")
    #     # exit()

    #     # 從這邊改modify整句
    #     # 1. 先切分句子
    #     sentences = sent_tokenize(current_text.text)
    #     current_text_clean = current_text.text.replace("<UNMODIFIABLE_start>", "").replace("<UNMODIFIABLE_end>", "")
    #     print(sentences[0:2])
    #     exit()

    #     # match the indices to modify with the sentence
    #     unmodified_set = set(range(len(current_text.words))) - indices_to_modify
    #     # print(f"Unmodified set: {unmodified_set}")
    #     unmodified_range = self.find_ranges(list(unmodified_set))

    #     # form the unmodified words with punctuation (range for current_text)
    #     unmodified_range_text = []
    #     current_text_list = self.split_text(current_text.text)
    #     # for sent in current_text_list:
    #     #     print(sent)
    #     #     print('------------------------')
    #     # print(current_text.words)
    #     # print(current_text_list)

    #     search_idx = 0
    #     for i in range(len(unmodified_range)):
    #         range_start, range_end = unmodified_range[i][0], unmodified_range[i][1]
    #         word_start, word_end = current_text.words[range_start], current_text.words[range_end]

    #         for j in range(search_idx, len(current_text_list)):
    #             if self.strip_punctuation(current_text_list[j]) == word_start:
    #                 start = j
    #             if self.strip_punctuation(current_text_list[j]) == word_end:
    #                 unmodified_range_text.append((start, j))
    #                 search_idx = j
    #                 break
    #         print(f"Unmodified range {word_start}, {word_end}")
    #         print(f"Unmodified range text: {current_text_list[start]}, {current_text_list[j]}")
    #     print()

    #     # split the text into sentences
    #     # sent_list = []
    #     current_text_sent_split = []
    #     last_end = None
    #     for range_idx in unmodified_range_text:
    #         range_start, range_end = range_idx[0], range_idx[1]

    #         if last_end is not None:
    #             modifiable_sent = " ".join(current_text_list[last_end+1:range_start])
    #             # print(f"Modifiable sentence: {modifiable_sent}")

    #             # sent_list.append([1, modifiable_sent])
    #             sents = sent_tokenize(modifiable_sent) # tokenize the text into sentences
    #             for sent in sents:
    #                 # print(sent)
    #                 # print('------------------------')
    #                 # sent = sent.replace("<UNMODIFIABLE_start>", "").replace("<UNMODIFIABLE_end>", "")
    #                 current_text_sent_split.append([1, sent])  # modifiable sentence

    #         unmodifiable_sent = " ".join(current_text_list[range_start:range_end+1])
    #         # sent_list.append([0, unmodifiable_sent])
    #         current_text_sent_split.append([0, unmodifiable_sent])  # unmodifiable sentence
    #         # print(unmodifiable_sent)
    #         # exit()
    #         last_end = range_end

    #     print(f"Current text sent split: {current_text_sent_split}")
    #     # for s in current_text_sent_split:
    #     #     if s[0] == 1:
    #     #         print(s)
    #     exit()
    #     # print()

    #     # transform the text
    #     transformed_result = []
    #     modified_sent = [(sent_idx, sent[1]) for sent_idx, sent in enumerate(current_text_sent_split) if sent[0] == 1] # sent[0] == 1: modifiable sentence
    #     print(f"Modified sentences: {modified_sent}")
    #     print(len(modified_sent), max(self.max_candidates // len(modified_sent), 1))

    #     for i in range(max(self.max_candidates // len(modified_sent), 1)): # at least 1 candidate for each sentence

    #         transformed_texts = self.transform_batch(modified_sent)

    #         print(transformed_texts)
    #         print('========================================================')

    #         tmp_split = copy.deepcopy(current_text_sent_split)
    #         for j in range(len(transformed_texts)):
    #             k = 0
    #             while tmp_split[k][0] == 0:
    #                 k += 1
    #             tmp_split[k][0] = 0  # Change to unmodifiable sentence
    #             result = copy.deepcopy(tmp_split)
    #             result[k] = [0, transformed_texts[j]]  # Replace & change to unmodifiable sentence
    #             transformed_result.append(" ".join([sent[1] for sent in result]))

    #     transformed_result = [AttackedText(text) for text in transformed_result]
    #     for text in transformed_result:
    #         print(f"Transformed text: {text.text}")
    #         print('------------------------')
    #     print(len(transformed_result))
    #     exit()
    #     return transformed_result
