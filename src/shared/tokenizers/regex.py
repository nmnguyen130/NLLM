"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Follows the algorithm of the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

Compared to BasicTokenizer:
- RegexTokenizer supports optional regex splitting.
- RegexTokenizer supports optional special tokens.
"""

import tqdm
import regex as re
from .base import Tokenizer, get_stats, merge

# Predefined split patterns used by GPT tokenizers
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class RegexTokenizer(Tokenizer):
    def __init__(self, pattern=None):
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {}
        self.inverse_special_tokens = {}

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # Use regex to split text into chunks
        text_chunks = re.findall(self.compiled_pattern, text)

        # Encode each chunk to bytes IDs
        token_ids = [list(ch.encode('utf-8')) for ch in text_chunks]

        # Merge most common bytes pairs
        merges = {}
        vocab = {i: bytes([i]) for i in range(256)}
        for i in tqdm.trange(num_merges, desc="Training tokenizer"):
            stats = {}
            for chunk_ids in token_ids:
                get_stats(chunk_ids, stats)  # Collect pair frequencies
            top_pair = max(stats, key=stats.get)  # Choose the most frequent pair
            idx = 256 + i
            token_ids = [merge(chunk_ids, top_pair, idx) for chunk_ids in token_ids]
            merges[top_pair] = idx
            vocab[idx] = vocab[top_pair[0]] + vocab[top_pair[1]]
            if verbose:
                print(f"Merge {i+1}/{num_merges}: {top_pair} -> {idx} ({vocab[idx]}) had {stats[top_pair]} occurrences")

        self.merges = merges
        self.vocab = vocab

    def register_special_tokens(self, special_tokens):
        """
        Register a dictionary of special tokens, e.g. {"<|endoftext|>": 100257}
        """
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def decode(self, ids):
        """
        Convert token IDs back into a UTF-8 string.
        Handles both normal and special tokens.
        """
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id: {idx}")
        return b"".join(part_bytes).decode("utf-8", errors="replace")
    
    def _encode_chunk(self, text_bytes):
        """
        Encode a single byte chunk using learned merges.
        """
        ids = list(text_bytes)
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids
    
    def encode_ordinary(self, text):
        """
        Encode the input string using regex splits, ignoring any special tokens.
        """
        text_chunks = re.findall(self.compiled_pattern, text)
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8")
            ids.extend(self._encode_chunk(chunk_bytes))
        return ids
    
    def encode(self, text, allowed_special="none_raise"):
        """
        Encode input text while handling special tokens if specified.
        - "all": allow all special tokens
        - "none": ignore all special tokens
        - "none_raise": raise error if special token is found
        - set: allow only a custom subset of special tokens
        """
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")

        if not special:
            return self.encode_ordinary(text)

        # Create regex to match special tokens and split text by them
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)

        # Encode each chunk separately
        ids = []
        for part in special_chunks:
            if part in special:
                ids.append(special[part])
            else:
                ids.extend(self.encode_ordinary(part))
        return ids
    
def main():
    # Sample text and vocab size
    file_name = "data/taylorswift.txt"
    with open(file_name, 'r', encoding='utf-8') as f:
        text = f.read()
    vocab_size = 1024

    # Initialize the tokenizer and train
    tokenizer = RegexTokenizer()
    tokenizer.train(text, vocab_size, verbose=False)

    input_text = "Reading practice to help you understand texts with everyday or job-related language. Texts include articles, travel guides, emails, adverts and reviews."

    # Encode and decode
    encoded = tokenizer.encode(input_text, allowed_special="all")
    print("\nTokens:")
    for token_id in encoded:
        token_bytes = tokenizer.vocab[token_id]
        try:
            token_str = token_bytes.decode("utf-8")
        except UnicodeDecodeError:
            token_str = str(token_bytes)
        print(f"{token_id}: {token_str}")
    print("\nEncoded:", encoded)

    decoded = tokenizer.decode(encoded)
    print("Decoded:", decoded)

if __name__ == "__main__":
    main()