"""Regex-based BPE tokenizer implementation."""

import tqdm
import regex as re
from typing import Dict, List, Set, Tuple, Union, Optional

from ..core.base import Tokenizer
from ..utils.text_processing import get_stats, merge, render_token


# Predefined split patterns used by GPT tokenizers
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class RegexTokenizer(Tokenizer):
    """A BPE tokenizer with regex-based pre-tokenization.
    
    Features:
    - Regex-based text splitting before BPE
    - Support for special tokens
    - Configurable tokenization patterns
    - Compatible with GPT-2/GPT-4 style tokenization
    """
    
    def __init__(self, pattern: Optional[str] = None):
        """Initialize the tokenizer with an optional regex pattern."""
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens: Dict[str, int] = {}
        self.inverse_special_tokens: Dict[int, str] = {}

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        """Train the tokenizer on the given text.
        
        Args:
            text: Training corpus as a string
            vocab_size: Desired vocabulary size (must be >= 256)
            verbose: Whether to print training progress debug info
        """
        if vocab_size < 256:
            raise ValueError("vocab_size must be >= 256")
        num_merges = vocab_size - 256
        
        # Use regex to split text into chunks
        text_chunks = re.findall(self.compiled_pattern, text)
        
        # Encode each chunk to bytes IDs
        token_ids = [list(chunk.encode('utf-8')) for chunk in text_chunks]
        
        # Initialize vocabulary and merges
        merges: Dict[Tuple[int, int], int] = {}
        vocab = {i: bytes([i]) for i in range(256)}
        
        # Perform BPE merges
        for i in tqdm.trange(num_merges, desc="Training tokenizer"):
            stats: Dict[Tuple[int, int], int] = {}
            
            # Count frequencies of all adjacent pairs
            for chunk_ids in token_ids:
                get_stats(chunk_ids, stats)
                
            if not stats:
                if verbose:
                    print(f"No more merges possible at {i+1}/{num_merges}")
                break
                
            # Find most frequent pair
            top_pair = max(stats, key=stats.get)
            idx = 256 + i
            
            # Update token IDs with the new merge
            token_ids = [merge(chunk_ids, top_pair, idx) for chunk_ids in token_ids]
            
            # Record the merge and update vocabulary
            merges[top_pair] = idx
            vocab[idx] = vocab[top_pair[0]] + vocab[top_pair[1]]
            
            if verbose:
                token_str = render_token(vocab[idx])
                print(f"Merge {i+1}/{num_merges}: {top_pair} -> {idx} ({token_str}) had {stats[top_pair]} occurrences")
        
        self.merges = merges
        self.vocab = vocab
    
    def register_special_tokens(self, special_tokens: Dict[str, int]) -> None:
        """Register special tokens that should be handled specially during encoding.
        
        Args:
            special_tokens: Dictionary mapping token strings to their IDs
        """
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}
    
    def _encode_chunk(self, text_bytes: bytes) -> List[int]:
        """Encode a single byte chunk using learned merges."""
        ids = list(text_bytes)
        
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float('inf')))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
            
        return ids
    
    def encode_ordinary(self, text: str) -> List[int]:
        """Encode text without handling special tokens."""
        text_chunks = self.compiled_pattern.findall(text)
        ids = []
        
        for chunk in text_chunks:
            chunk_bytes = chunk.encode('utf-8')
            ids.extend(self._encode_chunk(chunk_bytes))
            
        return ids
    
    def encode(
        self, 
        text: str, 
        allowed_special: Union[str, Set[str]] = "none_raise"
    ) -> List[int]:
        """Encode text with special token handling.
        
        Args:
            text: Input text to encode
            allowed_special: Controls how special tokens are handled:
                - "all": Allow all special tokens
                - "none": Ignore all special tokens
                - "none_raise": Raise error if special token is found
                - set: Allow only these special tokens
        """
        # Handle special tokens
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            for token in self.special_tokens:
                if token in text:
                    raise ValueError(
                        f"Special token '{token}' found in text. "
                        "Set allowed_special to 'all' or a set of allowed tokens."
                    )
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens.items() 
                     if k in allowed_special}
        else:
            raise ValueError(
                f"allowed_special={allowed_special} not understood. "
                "Expected 'all', 'none', 'none_raise', or a set of strings."
            )
        
        # Fast path: no special tokens to handle
        if not special:
            return self.encode_ordinary(text)
        
        # Create pattern to match special tokens
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)
        
        # Encode each chunk
        ids = []
        for part in special_chunks:
            if part in special:
                ids.append(special[part])
            else:
                ids.extend(self.encode_ordinary(part))
                
        return ids
    
    def decode(self, token_ids: List[int]) -> str:
        """Convert token IDs back to a string, handling special tokens."""
        part_bytes = []
        
        for token_id in token_ids:
            if token_id in self.vocab:
                part_bytes.append(self.vocab[token_id])
            elif token_id in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[token_id].encode('utf-8'))
            else:
                raise ValueError(f"Invalid token ID: {token_id}")
                
        return b''.join(part_bytes).decode('utf-8', errors='replace')
