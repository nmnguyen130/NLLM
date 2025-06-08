"""Text processing utilities for tokenizers."""

import unicodedata
from typing import Dict, List, Tuple, Optional


def get_stats(token_ids: List[int], counts: Optional[Dict[Tuple[int, int], int]] = None) -> Dict[Tuple[int, int], int]:
    """Count frequency of consecutive token pairs."""
    counts = {} if counts is None else counts
    for pair in zip(token_ids, token_ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(token_ids: List[int], target_pair: Tuple[int, int], new_token_id: int) -> List[int]:
    """Merge consecutive token pairs into a new token."""
    merged_ids = []
    i = 0
    while i < len(token_ids):
        if (i < len(token_ids) - 1) and (token_ids[i] == target_pair[0]) and (token_ids[i+1] == target_pair[1]):
            merged_ids.append(new_token_id)
            i += 2
        else:
            merged_ids.append(token_ids[i])
            i += 1
    return merged_ids


def replace_control_characters(text: str) -> str:
    """Replace control characters with escape sequences."""
    result = []
    for char in text:
        if unicodedata.category(char)[0] != 'C':
            result.append(char)
        else:
            result.append(f"\\u{ord(char):04x}")
    return "".join(result)


def render_token(token_bytes: bytes) -> str:
    """Convert token bytes to a display string."""
    decoded = token_bytes.decode('utf-8', errors='replace')
    return replace_control_characters(decoded)
