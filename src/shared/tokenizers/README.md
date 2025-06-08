# Tokenizers Module

A flexible and extensible tokenization library supporting multiple tokenization algorithms. This module provides various tokenizer implementations suitable for different natural language processing tasks.

## References:

Let's build the GPT Tokenizer: https://www.youtube.com/watch?v=zduSFxRajkE

## Features

- Multiple tokenizer implementations:
  - `BasicTokenizer`: Simple byte-pair encoding (BPE) tokenizer
  - `RegexTokenizer`: Advanced BPE with regex-based pre-tokenization
  - `HeapTokenizer`: Optimized BPE using heap and linked list for efficient training
- Common interface across all tokenizers
- Support for training custom tokenizers
- Save/load functionality for trained tokenizers
- Detailed token inspection and comparison utilities

## Installation

### From Source

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install in development mode:

   ```bash
   pip install -e .
   ```

   Or install just the tokenizers module:

   ```bash
   pip install -e src/shared/tokenizers
   ```

### Dependencies

- Python 3.7+
- Required packages (automatically installed with the module):
  - `regex` (for RegexTokenizer)
  - `tqdm` (for progress bars during training)

## Usage

### Basic Usage

```python
from shared.tokenizers import BasicTokenizer

# Initialize and train a tokenizer
tokenizer = BasicTokenizer()
tokenizer.train("your training text here", vocab_size=1000)

# Encode text
tokens = tokenizer.encode("Hello, world!")
print("Tokens:", tokens)

# Decode back to text
text = tokenizer.decode(tokens)
print("Decoded:", text)

# Save the tokenizer
tokenizer.save("my_tokenizer")

# Load a saved tokenizer
new_tokenizer = BasicTokenizer()
new_tokenizer.load("my_tokenizer.model")
```

### Using Different Tokenizers

```python
from shared.tokenizers import (
    BasicTokenizer,
    RegexTokenizer,
    HeapTokenizer,
    get_tokenizer_class
)

# Initialize specific tokenizer
tokenizer1 = BasicTokenizer()
tokenizer2 = RegexTokenizer()
tokenizer3 = HeapTokenizer()

# Or use the factory function
TokenizerClass = get_tokenizer_class('heap')  # 'basic', 'regex', or 'heap'
tokenizer = TokenizerClass()
```

### Training a Custom Tokenizer

```python
from shared.tokenizers import train_tokenizer

# Train a tokenizer with custom parameters
tokenizer = train_tokenizer(
    tokenizer_type='heap',  # 'basic', 'regex', or 'heap'
    text="your training text here",
    vocab_size=1000,
    save_path="custom_tokenizer",  # optional
    verbose=True
)
```

### Comparing Tokenizers

```python
from shared.tokenizers import compare_tokenizers, BasicTokenizer, RegexTokenizer

basic = BasicTokenizer()
regex = RegexTokenizer()

# Train tokenizers (if needed)
basic.train("sample text", vocab_size=300)
regex.train("sample text", vocab_size=300)

# Compare outputs
compare_tokenizers(basic, regex, "Text to compare tokenization")
```

## Command Line Interface

The module includes a command-line interface for quick testing and experimentation:

```bash
# Show help
python -m shared.tokenizers.main --help

# Train and compare all tokenizers
python -m shared.tokenizers.main --train

# Use a specific tokenizer
python -m shared.tokenizers.main --tokenizer heap --text "Your text here" --vocab-size 500

# Save a trained tokenizer
python -m shared.tokenizers.main --tokenizer basic --text "Training text" --save my_tokenizer
```

## Tokenizer Comparison

| Feature             | BasicTokenizer | RegexTokenizer | HeapTokenizer  |
| ------------------- | -------------- | -------------- | -------------- |
| Training Required   | Yes            | Yes            | Yes            |
| Pre-trained         | No             | No             | No             |
| Custom Vocabulary   | Yes            | Yes            | Yes            |
| Special Tokens      | No             | Yes            | Yes            |
| Training Efficiency | Low            | Medium         | High           |
| Best For            | Simple tasks   | Complex text   | Large datasets |

## Extending the Module

To create a custom tokenizer:

1. Create a new class that inherits from `Tokenizer` in `core/base.py`
2. Implement the required methods: `train()`, `encode()`, and `decode()`
3. Add your tokenizer to the `get_tokenizer_class()` function in `main.py`

## License

[Your License Here]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgements

- Inspired by various open-source tokenization libraries
- GPT-4 tokenizer implementation uses tiktoken by OpenAI
