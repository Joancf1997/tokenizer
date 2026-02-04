# BPE Tokenizer

A Python implementation of Byte Pair Encoding (BPE) tokenizer based on [Andrej Karpathy's tutorial](https://www.youtube.com/watch?v=zduSFxRajkE&t=4708s).

## Overview

This tokenizer uses the BPE algorithm to compress text by learning common byte pair patterns and merging them into new tokens. It's a fundamental building block used in many modern language models.

## Features

- ✅ **Train** on custom text corpus
- ✅ **Encode** text to token IDs
- ✅ **Decode** token IDs back to text
- ✅ **Save/Load** vocabulary and merges to/from files
- ✅ **UTF-8** support for multilingual text

## Installation

No external dependencies required! Just Python 3.6+.

```bash
# Clone or download this repository
git clone <your-repo-url>
cd tokenizer
```

## Quick Start

### 1. Training a New Tokenizer

```python
from bpe_tokenizer import BPETokenizer

# Create a tokenizer instance
tokenizer = BPETokenizer()

# Train on your text
training_text = """
The very name strikes fear and awe into the hearts of programmers worldwide. 
We all know we ought to "support Unicode" in our software.
"""

tokenizer.train(training_text, vocab_size=276)

# Save the trained tokenizer
tokenizer.save("vocab.json", "merges.json")
```

### 2. Using a Pre-trained Tokenizer

```python
from bpe_tokenizer import BPETokenizer

# Load a pre-trained tokenizer
tokenizer = BPETokenizer()
tokenizer.load("vocab.json", "merges.json")

# Encode text
text = "hello world"
token_ids = tokenizer.encode(text)
print(f"Encoded: {token_ids}")

# Decode back to text
decoded_text = tokenizer.decode(token_ids)
print(f"Decoded: {decoded_text}")
```

### 3. Running the Example

```bash
python bpe_tokenizer.py
```

This will:
- Train a tokenizer on sample Unicode text
- Test encoding/decoding
- Save `vocab.json` and `merges.json`
- Load and verify the saved tokenizer

## API Reference

### `BPETokenizer`

Main tokenizer class.

#### Constructor

```python
BPETokenizer(vocab=None, merges=None)
```

**Parameters:**
- `vocab` (dict, optional): Pre-existing vocabulary mapping token IDs to bytes
- `merges` (dict, optional): Pre-existing merge rules

#### Methods

##### `train(text, vocab_size=276)`

Train the tokenizer on the given text.

**Parameters:**
- `text` (str): Training text
- `vocab_size` (int): Desired vocabulary size (must be >= 256)

**Example:**
```python
tokenizer = BPETokenizer()
tokenizer.train("Hello, world!", vocab_size=300)
```

##### `encode(text) -> List[int]`

Encode text into a list of token IDs.

**Parameters:**
- `text` (str): Text to encode

**Returns:**
- List of token IDs (integers)

**Example:**
```python
token_ids = tokenizer.encode("Hello, world!")
# Output: [72, 101, 108, 108, 111, 44, 32, 119, 270, 108, 100, 33]
```

##### `decode(ids) -> str`

Decode token IDs back into text.

**Parameters:**
- `ids` (List[int]): List of token IDs

**Returns:**
- Decoded text string

**Example:**
```python
text = tokenizer.decode([72, 101, 108, 108, 111])
# Output: "Hello"
```

##### `save(vocab_path, merges_path)`

Save vocabulary and merges to JSON files.

**Parameters:**
- `vocab_path` (str): Path to save vocabulary JSON
- `merges_path` (str): Path to save merges JSON

**Example:**
```python
tokenizer.save("my_vocab.json", "my_merges.json")
```

##### `load(vocab_path, merges_path)`

Load vocabulary and merges from JSON files.

**Parameters:**
- `vocab_path` (str): Path to vocabulary JSON file
- `merges_path` (str): Path to merges JSON file

**Example:**
```python
tokenizer = BPETokenizer()
tokenizer.load("my_vocab.json", "my_merges.json")
```

## File Format

### `vocab.json`

Maps token IDs to their hexadecimal byte representation:

```json
{
  "0": "00",
  "1": "01",
  ...
  "256": "6520",
  "257": "f09f"
}
```

### `merges.json`

Maps byte pairs to their merged token ID:

```json
{
  "101,32": 256,
  "240,159": 257,
  "105,110": 258
}
```

## Understanding BPE

Byte Pair Encoding works by:

1. **Starting with bytes:** Text is encoded as UTF-8 bytes (tokens 0-255)
2. **Finding patterns:** The most frequent byte pairs are identified
3. **Merging pairs:** Common pairs are merged into new tokens (256+)
4. **Iterating:** Process repeats until desired vocabulary size is reached

This creates a vocabulary that efficiently represents common patterns in your training data!

## Example Output

```
merging (101, 32) into a new token 256
merging (240, 159) into a new token 257
merging (105, 110) into a new token 258
...

Training complete!
Original length: 608
Compressed length: 447
Compression ratio: 1.36X

Test encoding: 'hello world'
Encoded: [104, 101, 108, 108, 111, 32, 119, 270, 108, 100]
Decoded: 'hello world'
Match: True
```

## Resources

- [Andrej Karpathy's YouTube Tutorial](https://www.youtube.com/watch?v=zduSFxRajkE&t=4708s)
- [Tiktokenizer Web App](https://tiktokenizer.vercel.app/)
- [Unicode Programmer's Intro](https://www.reedbeta.com/blog/programmers-intro-to-unicode/)

## License

MIT License - Feel free to use and modify!