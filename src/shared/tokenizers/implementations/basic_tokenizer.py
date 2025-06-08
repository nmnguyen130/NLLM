"""Basic byte-level BPE tokenizer implementation."""

import tqdm
from typing import List

from ..core.base import Tokenizer
from ..utils.text_processing import get_stats, merge


class BasicTokenizer(Tokenizer):
    """A minimal byte-level BPE tokenizer.
    
    Implements a simplified version of the GPT tokenizer:
    - No regex splitting
    - No handling of special tokens by default
    - Pure byte-level BPE
    """
    
    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        """Train the BPE tokenizer on the given text.
        
        Args:
            text: Training corpus as a string
            vocab_size: Desired vocabulary size (must be >= 256)
            verbose: Whether to print training progress debug info
        """
        if vocab_size < 256:
            raise ValueError("vocab_size must be >= 256")    
        num_merges = vocab_size - 256

        # Convert text to UTF-8 bytes, then to a list of bytes values (0-255)
        text_bytes = text.encode('utf-8')
        token_ids = list(text_bytes)

        merges = {}
        vocab = {i: bytes([i]) for i in range(256)}

        for i in tqdm.trange(num_merges, desc="Training tokenizer"):
            # Count frequency of all adjacent pairs
            stats = get_stats(token_ids)
            if not stats:
                if verbose:
                    print(f"No more merges possible at {i+1}/{num_merges}")
                break
                
            # Choose the most frequent pair
            top_pair = max(stats, key=stats.get)
            # Assign a new ID to the merged pair
            idx = 256 + i
            # Replace occurrences of the pair with the new ID
            token_ids = merge(token_ids, top_pair, idx)
            # Record the merge
            merges[top_pair] = idx
            vocab[idx] = vocab[top_pair[0]] + vocab[top_pair[1]]

            if verbose:
                token_str = vocab[idx].decode('utf-8', errors="replace")
                print(f"Merge {i+1}/{num_merges}: {top_pair} -> {idx} ({token_str}) had {stats[top_pair]} occurrences")

        self.merges = merges
        self.vocab = vocab
    
    def encode(self, text: str) -> List[int]:
        """Encode text into token IDs using learned merges."""
        text_bytes = text.encode('utf-8')
        token_ids = list(text_bytes)

        while len(token_ids) >= 2:
            stats = get_stats(token_ids)
            # Find the pair with the lowest merge index (earliest learned)
            pair = min(stats, key=lambda p: self.merges.get(p, float('inf')))
            if pair not in self.merges:
                break  # No more merges available
            idx = self.merges[pair]
            token_ids = merge(token_ids, pair, idx)

        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to a string."""
        text_bytes = b"".join(self.vocab[idx] for idx in token_ids)
        return text_bytes.decode('utf-8', errors="replace")
