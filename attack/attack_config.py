UNMODIFIABLE_SET = [
    "Abstract:",
    "Abstract",
    "Introduction",
    "Related Work",
    "Method",
    "Experiments",
    "Discussion",
    "Conclusion",
    "Acknowledgement",
    "Reference",
    "UnmodifiableStart",
    "UnmodiableEnd",
    "<UnmodifiableStart>",
    "<UnmodiableEnd>",
    "Section",
]


# following promptbench/promptbench/prompt_attack/__init__.py
# constraint the # of possible transformations
attack_config = {
    "goal_function": {
        "query_budget": float("inf"),
        "top_k_wir": 50,
        "score_increase_threshold": 0.0,
    },
    "deepwordbug": {
        "levenshtein_edit_distance": 30,
    },
    "puncattack": {
        "semantic_distance": 0.8,
    },
    "bertattack": {
        "max_candidates": 15,  # due to the higher computational cost of using BERT
        "max_word_perturbed_percent": 1,
        "min_sentence_cos_sim": 0.8,
    },
    "textfooler": {
        "max_candidates": 15,
        "min_word_cos_sim": 0.7,
        "min_sentence_cos_sim": 0.8,
    },
    "styleadv": {
        "max_candidates": 50,
        "style": "bible",  # style list: ['bible', 'shakespeare', 'tweets', 'lyrics', 'poetry']
        "paraphraser_path": "attack/adv_reviewer/transformations/utils/style/inverseStyleTransfer",
    },
}
