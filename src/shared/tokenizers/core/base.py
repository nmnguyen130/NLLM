"""Base tokenizer class and core functionality."""

import unicodedata
from typing import Dict, List, Tuple, Optional

from ..utils.text_processing import get_stats, merge, replace_control_characters, render_token

class Tokenizer:
    """Base class for all tokenizers."""
    
    def __init__(self):
        self.merges: Dict[Tuple[int, int], int] = {}
        self.pattern: str = ""
        self.special_tokens: Dict[str, int] = {}
        self.vocab: Dict[int, bytes] = self._build_vocab()

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        """Train the tokenizer on input text."""
        raise NotImplementedError

    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs."""
        raise NotImplementedError

    def decode(self, token_ids: List[int]) -> str:
        """Convert token IDs back to text."""
        raise NotImplementedError
    
    def _build_vocab(self) -> Dict[int, bytes]:
        """Build vocabulary from base bytes and merges."""
        vocab = {i: bytes([i]) for i in range(256)}
        for (p1, p2), idx in self.merges.items():
            vocab[idx] = vocab[p1] + vocab[p2]
        for token_str, idx in self.special_tokens.items():
            vocab[idx] = token_str.encode('utf-8')
        return vocab
    
    def save(self, file_prefix: str) -> None:
        """Save tokenizer to files."""
        self._save_model(file_prefix)
        self._save_vocab(file_prefix)
    
    def _save_model(self, file_prefix: str) -> None:
        """Save model to .model file."""
        model_file = f"{file_prefix}.model"
        with open(model_file, 'w', encoding='utf-8') as f:
            f.write("SemTok v1\n")
            f.write(f"{self.pattern}\n")
            f.write(f"{len(self.special_tokens)}\n")
            for token_str, idx in self.special_tokens.items():
                f.write(f"{token_str} {idx}\n")
            for (p1, p2) in self.merges:
                f.write(f"{p1} {p2}\n")

    def _save_vocab(self, file_prefix: str) -> None:
        """Save human-readable vocabulary."""
        vocab_file = f"{file_prefix}.vocab"
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

    def load(self, model_file: str) -> None:
        """Load tokenizer from file."""
        if not model_file.endswith(".model"):
            raise ValueError("Expected .model file")
        
        try:
            merges = {}
            special_tokens = {}
            current_idx = 256

            with open(model_file, 'r', encoding='utf-8') as f:
                header = f.readline().strip()
                if header != "minbpe v1":
                    raise ValueError("Invalid model file format")
                
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
                    except ValueError as e:
                        raise ValueError(f"Invalid merge rule: {line.strip()}") from e

            self.merges = merges
            self.special_tokens = special_tokens
            self.vocab = self._build_vocab()
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Model file not found: {model_file}") from e
