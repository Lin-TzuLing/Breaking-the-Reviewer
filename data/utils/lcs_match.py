from nltk.tokenize import sent_tokenize, word_tokenize


def find_lcs(a, b):
    """
    Find the Longest Common Substring (LCS) between two strings.
    Returns the LCS and its start and end positions in `a``.
    """
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    length, end_pos = 0, 0  # Initialize LCS length and end position

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > length:
                    length = dp[i][j]
                    end_pos = i
    return a[end_pos - length : end_pos], end_pos - length, end_pos


def find_enclosing_range(ranges, start, end):
    """
    Given a list of ranges `ranges = [(start1, end1), (start2, end2), ...]`,
    find the first interval that completely contains `start, end`.
    """
    overlapping = [(s, e) for s, e in ranges if not (e < start or s > end)]
    return (overlapping[0][0], overlapping[-1][-1])


def add_modifiable_markers(string, patterns, lcs_split_level="char"):
    """
    Find each pattern and its LCS in `string`, and add markers before and after them.
    """
    if lcs_split_level == "char":
        modifiable_string = string
        # start_token, end_token = "<UNMODIFIABLE_start>", "<UNMODIFIABLE_end>"
        start_token, end_token = "<UnmodifiableStart>", "<UnmodifiableEnd>"
    elif lcs_split_level == "word" or lcs_split_level == "sent":
        modifiable_string = word_tokenize(string)
        patterns = [word_tokenize(pattern) for pattern in patterns]
        start_token, end_token = ["<UnmodifiableStart>"], ["<UnmodifiableEnd>"]

        sent_split = sent_tokenize(string)
        word_pointer = 0  # Track current word index

        sent_idx = []
        for sent in sent_split:
            sent_word_count = len(word_tokenize(sent))  # Number of words in sentence
            start_idx = word_pointer
            end_idx = word_pointer + sent_word_count
            sent_idx.append((start_idx, end_idx))
            word_pointer += sent_word_count
    else:
        raise ValueError("Invalid lcs_split_level. Choose 'char', 'word', or 'sent'.")

    insert_idx_list = []

    for pattern in patterns:
        if lcs_split_level == "sent":
            lcs, start, end = find_lcs(modifiable_string, pattern)
            # print(f"Pattern: {pattern}")
            # print(f"Longest Common Subsequence: {lcs}")
            # print(f"Start: {start}, End: {end}")
            # print("=======================================================")
            if lcs:
                start_idx, end_idx = find_enclosing_range(sent_idx, start, end)
                insert_idx_list.append((start_idx, end_idx))
        else:
            lcs, start, end = find_lcs(modifiable_string, pattern)
            # print(f"Pattern: {pattern}")
            # print(f"Longest Common Subsequence: {lcs}")
            # print(f"Start: {start}, End: {end}")
            # print("=======================================================")
            if lcs:
                insert_idx_list.append((start, end))

    # Merge overlapping or adjacent ranges to prevent multiple insertions
    merged_ranges = []
    for start, end in sorted(insert_idx_list):
        if merged_ranges and start <= merged_ranges[-1][1]:  # Overlapping or adjacent
            merged_ranges[-1] = (merged_ranges[-1][0], max(merged_ranges[-1][1], end))
        else:
            merged_ranges.append((start, end))

    # Insert markers
    offset = 0
    for start, end in merged_ranges:
        start += offset
        end += offset

        modifiable_string = (
            modifiable_string[:start]
            + end_token
            + modifiable_string[start:end]
            + start_token
            + modifiable_string[end:]
        )
        offset += len(end_token) + len(start_token)

    # Add markers around the entire string
    modifiable_string = start_token + modifiable_string + end_token

    if lcs_split_level != "char":
        modifiable_string = " ".join(modifiable_string)

    return modifiable_string
