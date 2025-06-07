import unicodedata
from typing import Dict, List, Tuple, Optional

def get_stats(token_ids: List[int], counts: Optional[Dict[Tuple[int, int], int]] = None) -> Dict[Tuple[int, int], int]:
    """
    Count the frequency of consecutive token pairs in the list `token_ids`.
    Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    """
    counts = {} if counts is None else counts
    for pair in zip(token_ids, token_ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(token_ids: List[int], target_pair: Tuple[int, int], new_token_id: int) -> List[int]:
    """
    Replace all consecutive occurrences of `target_pair` with `new_token_id`.
    Example: [1, 2, 3, 1, 2], target_pair=(1, 2), new_token_id=4 -> [4, 3, 4]
    """
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
    """
    Replace Unicode control characters in the string with escape sequences.
    Control characters can break output formatting or display.
    """
    result = []
    for char in text:
        if unicodedata.category(char)[0] != 'C':
            result.append(char)
        else:
            result.append(f"\\u{ord(char):04x}")

    return "".join(result)

def render_token(token_bytes: bytes) -> str:
    """
    Decode the byte-level token into a string, replacing control characters.
    """
    decoded = token_bytes.decode('utf-8', errors='replace')
    return replace_control_characters(decoded)

# The Base Tokenizer
class Tokenizer:
    """Base class for tokenizers."""
    def __init__(self):
        # Default values: vocab_size=256 (all bytes), no merges, no pattern
        self.merges: Dict[Tuple[int, int], int] = {}        # (int, int) -> int; token merge rules
        self.pattern: str = ""                              # optional regex pattern (used by subclasses)
        self.special_tokens: Dict[str, int] = {}            # str -> int; e.g., {'<|endoftext|>': 100257}
        self.vocab: Dict[int, bytes] = self._build_vocab()  # int -> bytes; full vocabulary

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        """Train the tokenizer on input text (to be implemented by subclass)."""
        raise NotImplementedError

    def encode(self, text: str) -> List[int]:
        """Convert string to a list of token IDs (to be implemented)."""
        raise NotImplementedError

    def decode(self, token_ids: List[int]) -> str:
        """Convert a list of token IDs back to a string (to be implemented)."""
        raise NotImplementedError
    
    def _build_vocab(self) -> Dict[int, bytes]:
        """
        Construct vocabulary from base bytes, merges, and special tokens.
        This determines how tokens are decoded.
        """
        vocab = {i: bytes([i]) for i in range(256)}  # 1-byte tokens
        for (p1, p2), idx in self.merges.items():
            vocab[idx] = vocab[p1] + vocab[p2]
        for token_str, idx in self.special_tokens.items():
            vocab[idx] = token_str.encode('utf-8')
        return vocab
    
    def save(self, file_prefix):
        """
        Save the tokenizer model and human-readable vocabulary to disk.
        Files:
          - {file_prefix}.model: for loading the tokenizer
          - {file_prefix}.vocab: for human inspection (not used for loading)
        """
        self._save_model(file_prefix)
        self._save_vocab(file_prefix)
    
    def _save_model(self, file_prefix):
        """Save tokenizer model to {file_prefix}.model."""
        model_file = file_prefix + ".model"
        with open(model_file, 'w', encoding='utf-8') as f:
            f.write("minbpe v1\n")
            f.write(f"{self.pattern}\n")
            f.write(f"{len(self.special_tokens)}\n")
            for token_str, idx in self.special_tokens.items():
                f.write(f"{token_str} {idx}\n")
            for (p1, p2) in self.merges:
                f.write(f"{p1} {p2}\n")

    def _save_vocab(self, file_prefix):
        """Save human-readable vocabulary to {file_prefix}.vocab."""
        vocab_file = file_prefix + ".vocab"
        inverse_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, 'w', encoding='utf-8') as f:
            for idx, token_bytes in self.vocab.items():
                token_str = render_token(token_bytes)
                if idx in inverse_merges:
                    p1, p2 = inverse_merges[idx]
                    s1 = render_token(self.vocab[p1])
                    s2 = render_token(self.vocab[p2])
                    f.write(f"[{s1} {s2}] -> [{token_str}] {idx}\n")
                else:
                    f.write(f"[{token_str}] {idx}\n")

    def load(self, model_file):
        """
        Load the tokenizer from a .model file.
        Reconstructs merge rules, special tokens, and builds the vocabulary.
        """
        if not model_file.endswith(".model"):
            raise AssertionError(f"Expected .model file, got {model_file}")
        
        try:
            merges = {}
            special_tokens = {}
            current_idx = 256

            with open(model_file, 'r', encoding='utf-8') as f:
                header = f.readline().strip()
                if header != "minbpe v1":
                    raise AssertionError(f"Invalid model file: {model_file}")
                
                self.pattern = f.readline().strip()
                num_special = int(f.readline().strip())

                for _ in range(num_special):
                    line = f.readline().strip()
                    if not line:
                        raise ValueError("Invalid special token line")
                    token, token_id = line.split()
                    special_tokens[token] = int(token_id)

                for line in f:
                    try:
                        p1, p2 = map(int, line.strip().split())
                        merges[(p1, p2)] = current_idx
                        current_idx += 1
                    except ValueError:
                        raise ValueError(f"Invalid merge rule: {line.strip()}")

            self.merges = merges
            self.special_tokens = special_tokens
            self.vocab = self._build_vocab()
        except:
            raise FileNotFoundError(f"Model file not found: {model_file}")