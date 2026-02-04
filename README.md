# BPE Tokenizer

A clean Python implementation of Byte Pair Encoding (BPE) tokenizer, inspired by and based on [Andrej Karpathy's educational tutorial](https://www.youtube.com/watch?v=zduSFxRajkE&t=4708s).

## What is this?

This project implements a **Byte Pair Encoding (BPE) tokenizer** from scratch - the same compression algorithm used in modern language models like GPT-2, GPT-3, and many others. It started as a learning exercise following Andrej Karpathy's excellent tutorial on building tokenizers, and has been packaged into a reusable Python class.

### Why BPE Matters

Tokenization is a critical first step in natural language processing. BPE strikes a balance between character-level (too granular) and word-level (too sparse) tokenization by learning frequent byte patterns from your data and merging them into single tokens. This creates an efficient vocabulary that captures common words and subwords.

## Features

- 🎓 **Educational**: Clean, well-documented code following Andrej's teaching approach
- 🚀 **Complete**: Train, encode, decode, save, and load tokenizers
- 🌍 **UTF-8 Ready**: Full Unicode support for multilingual text
- 📦 **Zero Dependencies**: Pure Python 3.6+ implementation
- 💾 **Persistent**: Export and reuse trained vocabularies

## Quick Start

### Training Your Own Tokenizer

```python
from bpe_tokenizer import BPETokenizer

# Create and train on your text
tokenizer = BPETokenizer()
tokenizer.train("Your training text here...", vocab_size=300)

# Save for later use
tokenizer.save("vocab.json", "merges.json")
```

### Using a Trained Tokenizer

```python
from bpe_tokenizer import BPETokenizer

# Load pre-trained tokenizer
tokenizer = BPETokenizer()
tokenizer.load("vocab.json", "merges.json")

# Encode and decode
tokens = tokenizer.encode("hello world")
text = tokenizer.decode(tokens)
```

### Run the Example

```bash
python bpe_tokenizer.py
```

This demonstrates training on Unicode text, encoding/decoding, and persistence.

## What You Get

After training, you'll have:
- **`vocab.json`**: Mapping of token IDs to their byte representations
- **`merges.json`**: The learned merge rules (which byte pairs to combine)
- A working tokenizer that compresses text efficiently

## How It Works

Following Andrej's explanation, the algorithm is elegant:

1. Start with all 256 possible bytes as base tokens (0-255)
2. Find the most frequent adjacent pair of tokens in your text
3. Merge that pair into a new token (256+)
4. Repeat until you reach your target vocabulary size

The result: common patterns (like "ing", "the", "er") become single tokens, yielding better compression than raw bytes.

## Example Output

```
Training complete!
Original length: 608
Compressed length: 447
Compression ratio: 1.36X
```

## Key Methods

| Method | Purpose |
|--------|---------|
| `train(text, vocab_size)` | Learn BPE merges from training text |
| `encode(text)` | Convert text → token IDs |
| `decode(ids)` | Convert token IDs → text |
| `save(vocab_path, merges_path)` | Persist the tokenizer |
| `load(vocab_path, merges_path)` | Reload a saved tokenizer |

## Acknowledgments

This implementation is built following **[Andrej Karpathy's](https://karpathy.ai/)** excellent educational content on building tokenizers from scratch. His tutorial demystifies how modern language models process text and provides the foundation for this code.

## Resources

- 📺 [Andrej Karpathy's YouTube Tutorial](https://www.youtube.com/watch?v=zduSFxRajkE&t=4708s) - The inspiration for this project
- 🔧 [Tiktokenizer](https://tiktokenizer.vercel.app/) - Interactive tokenizer visualization
- 📖 [Unicode Programmer's Intro](https://www.reedbeta.com/blog/programmers-intro-to-unicode/) - Understanding text encoding

## License

MIT License - Feel free to learn from, use, and modify!