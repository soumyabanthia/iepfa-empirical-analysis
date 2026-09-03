from typing import Tuple, Optional


def levenshtein_ratio(s1: str, s2: str) -> float:
    s1, s2 = s1.upper().strip(), s2.upper().strip()
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    len1, len2 = len(s1), len(s2)
    matrix = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    for i in range(len1 + 1):
        matrix[i][0] = i
    for j in range(len2 + 1):
        matrix[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            matrix[i][j] = min(
                matrix[i - 1][j] + 1,
                matrix[i][j - 1] + 1,
                matrix[i - 1][j - 1] + cost,
            )

    dist = matrix[len1][len2]
    max_len = max(len1, len2)
    return round((1.0 - (dist / max_len)), 3)


def soundex_key(name: str) -> str:
    name = "".join(c for c in name.upper() if c.isalpha())
    if not name:
        return "0000"

    mapping = {
        'B': '1', 'F': '1', 'P': '1', 'V': '1',
        'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
        'D': '3', 'T': '3',
        'L': '4',
        'M': '5', 'N': '5',
        'R': '6'
    }

    first_letter = name[0]
    encoded = [first_letter]

    prev_code = mapping.get(first_letter, '0')
    for char in name[1:]:
        code = mapping.get(char, '0')
        if code != '0' and code != prev_code:
            encoded.append(code)
        prev_code = code

    soundex_str = "".join(encoded).replace('0', '')
    return (soundex_str + "0000")[:4]


def evaluate_identity_match(
    claimant_pan: str,
    claimant_name: str,
    record_pan: Optional[str],
    record_name: str
) -> Tuple[bool, float, str]:
    claimant_pan = claimant_pan.strip().upper()
    claimant_name = claimant_name.strip().upper()
    record_name = record_name.strip().upper()

    if record_pan and record_pan.strip().upper() == claimant_pan:
        return True, 100.0, "EXACT_PAN_MATCH"

    sim_ratio = levenshtein_ratio(claimant_name, record_name)
    soundex_claimant = soundex_key(claimant_name)
    soundex_record = soundex_key(record_name)

    if sim_ratio >= 0.90:
        return True, round(sim_ratio * 100, 1), "HIGH_NAME_SIMILARITY"

    if soundex_claimant == soundex_record and sim_ratio >= 0.70:
        return True, round(sim_ratio * 100, 1), "PHONETIC_SOUNDEX_MATCH"

    claimant_tokens = set(claimant_name.split())
    record_tokens = set(record_name.split())
    common_tokens = claimant_tokens.intersection(record_tokens)

    if len(common_tokens) >= 1 and any(len(t) > 3 for t in common_tokens):
        return True, 75.0, "POTENTIAL_LEGAL_HEIR_SURNAME_MATCH"

    return False, 0.0, "NO_MATCH"

