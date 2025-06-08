"""
Tokenizers Demo and Usage Examples

This module demonstrates how to use the tokenizers package and serves as
an example of how to integrate it into other projects.

Example usage from another module:
    ```python
    from shared.tokenizers import BasicTokenizer, RegexTokenizer, HeapTokenizer, GPT4Tokenizer
    
    # Initialize and train a tokenizer
    tokenizer = BasicTokenizer()
    tokenizer.train("your text here", vocab_size=1000)
    
    # Encode text
    tokens = tokenizer.encode("text to encode")
    
    # Decode back to text
    text = tokenizer.decode(tokens)
    
    # Save and load
    tokenizer.save("my_tokenizer")
    tokenizer.load("my_tokenizer.model")
    ```
"""

import argparse
from typing import List, Dict, Any, Type, Union

from . import BasicTokenizer, RegexTokenizer, HeapTokenizer


def get_tokenizer_class(name: str) -> Type[Union[BasicTokenizer, RegexTokenizer, HeapTokenizer]]:
    """Get tokenizer class by name."""
    tokenizers = {
        'basic': BasicTokenizer,
        'regex': RegexTokenizer,
        'heap': HeapTokenizer,
    }
    return tokenizers.get(name.lower(), BasicTokenizer)


def train_tokenizer(
    tokenizer_type: str,
    text: str,
    vocab_size: int = 1000,
    save_path: str = None,
    verbose: bool = False
) -> Union[BasicTokenizer, RegexTokenizer, HeapTokenizer]:
    """Train a tokenizer on the given text.
    
    Args:
        tokenizer_type: Type of tokenizer ('basic', 'regex', 'heap', 'gpt4')
        text: Training text
        vocab_size: Size of the vocabulary
        save_path: If provided, save the trained tokenizer to this path
        verbose: Whether to print training progress
        
    Returns:
        Trained tokenizer instance
    """
    tokenizer_class = get_tokenizer_class(tokenizer_type)
    tokenizer = tokenizer_class()
    
    if tokenizer_type.lower() == 'gpt4':
        print("GPT4Tokenizer is pre-trained, skipping training")
    else:
        tokenizer.train(text, vocab_size=vocab_size, verbose=verbose)
    
    if save_path:
        tokenizer.save(save_path)
        print(f"Saved {tokenizer_type} tokenizer to {save_path}")
    
    return tokenizer


def encode_text(
    tokenizer: Union[BasicTokenizer, RegexTokenizer, HeapTokenizer],
    text: str,
    show_tokens: bool = False
) -> None:
    """Encode text and optionally show token details."""
    token_ids = tokenizer.encode(text)
    print(f"Encoded IDs: {token_ids}")
    
    if show_tokens:
        print("\nToken Details:")
        for i, tid in enumerate(token_ids):
            token_bytes = tokenizer.vocab.get(tid, b'<UNK>')
            try:
                token_str = token_bytes.decode('utf-8', errors='replace')
            except:
                token_str = str(token_bytes)
            print(f"  {i:3d}: ID={tid:5d}, Token='{token_str}'")
    
    # Verify round-trip
    decoded = tokenizer.decode(token_ids)
    print(f"\nDecoded: {decoded}")
    print(f"Match: {decoded == text}")


def compare_tokenizers(
    tokenizer1: Union[BasicTokenizer, RegexTokenizer, HeapTokenizer],
    tokenizer2: Union[BasicTokenizer, RegexTokenizer, HeapTokenizer],
    text: str
) -> None:
    """Compare the output of two tokenizers on the same text."""
    ids1 = tokenizer1.encode(text)
    ids2 = tokenizer2.encode(text)
    
    def get_token_details(tokenizer, token_ids):
        details = []
        for tid in token_ids:
            token_bytes = tokenizer.vocab.get(tid, b'<UNK>')
            try:
                token_str = token_bytes.decode('utf-8', errors='replace')
            except:
                token_str = str(token_bytes)
            details.append(f"{tid}:{token_str}")
        return details
    
    print(f"\nComparing tokenizers on: {text}")
    print(f"{tokenizer1.__class__.__name__}: {ids1}")
    print(f"{tokenizer2.__class__.__name__}: {ids2}")
    
    print("\nToken details:")
    print(f"{tokenizer1.__class__.__name__}: {' | '.join(get_token_details(tokenizer1, ids1))}")
    print(f"{tokenizer2.__class__.__name__}: {' | '.join(get_token_details(tokenizer2, ids2))}")
    
    # Compare the tokenized output
    decoded1 = tokenizer1.decode(ids1)
    decoded2 = tokenizer2.decode(ids2)
    
    print("\nDecoded:")
    print(f"{tokenizer1.__class__.__name__}: {decoded1}")
    print(f"{tokenizer2.__class__.__name__}: {decoded2}")
    
    if decoded1 == decoded2:
        print("\n✅ Decoded outputs match")
    else:
        print("\n❌ Decoded outputs differ")


def main():
    # Example usage
    sample_text = "the theater is there for those who think they're thinkers"
    
    # Train and compare different tokenizers
    basic = train_tokenizer('basic', sample_text, vocab_size=300, verbose=True, save_path="basic_tokenizer")
    regex = train_tokenizer('regex', sample_text, vocab_size=300, verbose=True, save_path="regex_tokenizer")
    heap = train_tokenizer('heap', sample_text, vocab_size=300, verbose=True, save_path="heap_tokenizer")
    
    # GPT-4 tokenizer is pre-trained
    gpt4 = train_tokenizer('gpt4', sample_text)
    
    # Encode some text with each tokenizer
    test_text = "for those who say they're for the theater"
    
    print("\n" + "="*50)
    print("BasicTokenizer:")
    encode_text(basic, test_text, show_tokens=True)
    
    print("\n" + "="*50)
    print("RegexTokenizer:")
    encode_text(regex, test_text, show_tokens=True)
    
    print("\n" + "="*50)
    print("HeapTokenizer:")
    encode_text(heap, test_text, show_tokens=True)
    
    print("\n" + "="*50)
    print("GPT4Tokenizer:")
    encode_text(gpt4, test_text, show_tokens=True)
    
    # Compare tokenizers
    print("\n" + "="*50)
    print("Comparing Basic vs Heap tokenizers:")
    compare_tokenizers(basic, heap, test_text)
    
    print("\n" + "="*50)
    print("Comparing Regex vs GPT-4 tokenizers:")
    compare_tokenizers(regex, gpt4, test_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tokenizers Demo')
    parser.add_argument('--train', action='store_true', help='Train and compare all tokenizers')
    parser.add_argument('--text', type=str, default="Hello, world! This is a test.", 
                        help='Text to tokenize')
    parser.add_argument('--tokenizer', type=str, default='heap', 
                        choices=['basic', 'regex', 'heap', 'gpt4'],
                        help='Tokenizer type to use')
    parser.add_argument('--vocab-size', type=int, default=300,
                        help='Vocabulary size (for trainable tokenizers)')
    parser.add_argument('--save', type=str, help='Save tokenizer to this path')
    
    args = parser.parse_args()
    
    if args.train:
        main()
    else:
        # Simple tokenization example
        tokenizer_class = get_tokenizer_class(args.tokenizer)
        tokenizer = tokenizer_class()
        
        if args.tokenizer != 'gpt4':
            print(f"Training {args.tokenizer} tokenizer...")
            tokenizer.train(args.text, vocab_size=args.vocab_size)
            if args.save:
                tokenizer.save(args.save)
        else:
            print("Using pre-trained GPT-4 tokenizer")
        
        encode_text(tokenizer, args.text, show_tokens=True)