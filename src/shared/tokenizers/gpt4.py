"""
Implements the GPT-4 Tokenizer as a lightweight wrapper around RegexTokenizer.
This is a **pretrained** tokenizer that mimics `tiktoken`'s `cl100k_base`.
"""

import tiktoken
from .regex import RegexTokenizer

def bpe(mergeable_ranks, token, max_rank):
    """
    Apply BPE merges to reconstruct the merge path (merge forest) of a token.
    This function finds the most frequent merge (lowest rank) until no more merges are allowed.
    """
    parts = [bytes([b]) for b in token]
    while True:
        min_idx = None
        min_rank = None
        for i, pair in enumerate(zip(parts[:-1], parts[1:])):
            rank = mergeable_ranks.get(pair[0] + pair[1])
            if rank is not None and (min_rank is None or rank < min_rank):
                min_idx = i
                min_rank = rank
        if min_rank is None or (max_rank is not None and min_rank >= max_rank):
            break
        parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2:]
    return parts

def recover_merges(mergeable_ranks):
    """
    Reverse engineer the merge rules from the final merged tokens in `mergeable_ranks`.
    This reconstructs the merge tree as (token_id_1, token_id_2) -> new_token_id.
    """
    merges = {}
    for token, rank in mergeable_ranks.items():
        if len(token) == 1:
            continue  # Skip single-byte tokens
        pair = tuple(bpe(mergeable_ranks, token, max_rank=rank))
        assert len(pair) == 2, f"Invalid merge pair: {pair}"
        ix0 = mergeable_ranks[pair[0]]
        ix1 = mergeable_ranks[pair[1]]
        merges[(ix0, ix1)] = rank
    return merges

# Pattern used for splitting input text, mimicking GPT-4's tokenizer
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

# GPT-4 special tokens and their IDs
GPT4_SPECIAL_TOKENS = {
    '<|endoftext|>': 100257,
    '<|fim_prefix|>': 100258,
    '<|fim_middle|>': 100259,
    '<|fim_suffix|>': 100260,
    '<|endofprompt|>': 100276
}

class GPT4Tokenizer(RegexTokenizer):
    """Wrapper over RegexTokenizer that replicates GPT-4's tokenizer behavior."""
    def __init__(self):
        super().__init__(pattern=GPT4_SPLIT_PATTERN)

        # Load tiktoken's pretrained base tokenizer
        enc = tiktoken.get_encoding("cl100k_base")
        mergeable_ranks = enc._mergeable_ranks

        # Reconstruct the GPT-4 merge rules
        self.merges = recover_merges(mergeable_ranks)

        # Rebuild vocab from byte and merged token IDs
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (a, b), idx in self.merges.items():
            vocab[idx] = vocab[a] + vocab[b]
        self.vocab = vocab

        # Handle GPT-4's byte shuffle (tiktoken uses a weird permuted byte mapping)
        self.byte_shuffle = {i: mergeable_ranks[bytes([i])] for i in range(256)}
        self.inverse_byte_shuffle = {v: k for k, v in self.byte_shuffle.items()}

        # Register special tokens
        self.register_special_tokens(GPT4_SPECIAL_TOKENS)

    def _encode_chunk(self, text_bytes):
        # Apply GPT-4's byte shuffle before encoding
        text_bytes = bytes(self.byte_shuffle[b] for b in text_bytes)
        ids = super()._encode_chunk(text_bytes)
        return ids
    
    def decode(self, token_ids):
        # Reconstruct original bytes and undo byte shuffle
        text_bytes = b"".join(self.vocab[idx] for idx in token_ids)
        text_bytes = bytes(self.inverse_byte_shuffle[b] for b in text_bytes)
        return text_bytes.decode('utf-8', errors="replace")
    
    def train(self, text, vocab_size, verbose=False):
        raise NotImplementedError("GPT4Tokenizer is pretrained and cannot be trained.")

    def save(self, file_prefix):
        raise NotImplementedError("GPT4Tokenizer cannot be saved.")

    def load(self, model_file):
        raise NotImplementedError("GPT4Tokenizer cannot be loaded.")
    
    def save_vocab(self, vocab_file):
        """
        Write the tokenizer vocabulary to a file in a human-readable format.
        Mostly for debugging and visualization.
        """
        from .base import render_token
        vocab = {idx: bytes([self.inverse_byte_shuffle[idx]]) for idx in range(256)}
        for (a, b), idx in self.merges.items():
            vocab[idx] = vocab[a] + vocab[b]
        inverse_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, 'w', encoding='utf-8') as f:
            for idx, token in vocab.items():
                s = render_token(token)
                if idx in inverse_merges:
                    idx0, idx1 = inverse_merges[idx]
                    s0 = render_token(vocab[idx0])
                    s1 = render_token(vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    f.write(f"[{s}] {idx}\n")

def main():
    # Initialize tokenizer
    tokenizer = GPT4Tokenizer()

    # Sample inputs
    texts = [
        "Hello world!",
        "GPT-4 is awesome 🚀",
        "Let's test tokenization.",
        "The quick brown fox jumps over the lazy dog."
    ]

    for text in texts:
        print(f"\nInput: {text}")
        token_ids = tokenizer.encode(text)
        print(f"Token IDs: {token_ids}")
        decoded = tokenizer.decode(token_ids)
        print(f"Decoded: {decoded}")

if __name__ == "__main__":
    main()
