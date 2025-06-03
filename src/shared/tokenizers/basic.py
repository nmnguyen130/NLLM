"""
Minimal byte-level Byte Pair Encoding (BPE) tokenizer.

Implements a simplified version of the GPT tokenizer:
- No regex splitting
- No handling of special tokens
"""

import tqdm
from typing import List
from .base import Tokenizer, get_stats, merge

class BasicTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        """
        Train the BPE tokenizer from raw text.
        Args:
            text (str): training corpus
            vocab_size (int): desired size of final vocabulary (must be >= 256)
            verbose (bool): print debug info
        """
        assert vocab_size >= 256, "vocab_size must be >= 256"
        num_merges = vocab_size - 256

        # Convert text to UTF-8 bytes, then to a list of bytes values (0-255)
        text_bytes = text.encode('utf-8')
        token_ids = list(text_bytes)

        # Prepare initial vocab and merge rules
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
                s = vocab[idx].decode('utf-8', errors="replace")
                print(f"Merge {i+1}/{num_merges}: {top_pair} -> {idx} ({s}) had {stats[top_pair]} occurrences")

        self.merges = merges
        self.vocab = vocab
    
    def encode(self, text: str) -> List[int]:
        """
        Encode a string into a list of token IDs using trained merges.
        """
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
        """
        Decode a list of token IDs back to a string.
        """
        text_bytes = b"".join(self.vocab[idx] for idx in token_ids)
        return text_bytes.decode('utf-8', errors="replace")
    
def main():
    file_name = "data/taylorswift.txt"
    with open(file_name, 'r', encoding='utf-8') as f:
        text = f.read()
    vocab_size = 1000

    # Initialize and train the tokenizer
    tokenizer = BasicTokenizer()
    tokenizer.train(text, vocab_size, verbose=False)
    tokenizer.save("taylorswift")

    # Encode a sample string
    input_text = "Reading practice to help you understand texts with everyday or job-related language. Texts include articles, travel guides, emails, adverts and reviews."
    # input_text = "Thực hành đọc để giúp bạn hiểu các văn bản bằng ngôn ngữ hàng ngày hoặc liên quan đến công việc. Văn bản bao gồm các bài viết, hướng dẫn du lịch, email, quảng cáo và đánh giá."
    encoded = tokenizer.encode(input_text)
    print("\nTokens:")
    for token_id in encoded:
        token_bytes = tokenizer.vocab[token_id]
        try:
            token_str = token_bytes.decode("utf-8")
        except UnicodeDecodeError:
            token_str = str(token_bytes)
        print(f"{token_id}: {token_str}")
    print("\nEncoded:", encoded)

    # Decode the token IDs back to string
    decoded = tokenizer.decode(encoded)
    print("Decoded:", decoded)

if __name__ == "__main__":
    main()