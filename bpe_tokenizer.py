"""
BPE (Byte Pair Encoding) Tokenizer

A simple implementation of Byte Pair Encoding tokenizer based on the Andrej Karpathy tutorial.
This tokenizer can encode text into tokens and decode tokens back to text.
"""

import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class BPETokenizer:
    """
    Byte Pair Encoding Tokenizer
    
    This tokenizer uses BPE algorithm to compress text by learning common byte pair patterns
    and merging them into new tokens.
    """
    
    def __init__(self, vocab: Optional[Dict[int, bytes]] = None, merges: Optional[Dict[Tuple[int, int], int]] = None):
        """
        Initialize the BPE tokenizer.
        
        Args:
            vocab: Dictionary mapping token IDs to their byte representations
            merges: Dictionary mapping byte pairs to their merged token ID
        """
        self.vocab = vocab if vocab is not None else {}
        self.merges = merges if merges is not None else {}
    
    def train(self, text: str, vocab_size: int = 276) -> None:
        """
        Train the tokenizer on the given text.
        
        Args:
            text: Training text
            vocab_size: Desired vocabulary size (must be >= 256)
        """
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256")
        
        # Encode text to UTF-8 bytes
        tokens = list(text.encode("utf-8"))
        num_merges = vocab_size - 256
        
        # Initialize vocabulary with all single bytes
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        self.merges = {}
        
        # Create a working copy of tokens
        ids = list(tokens)
        
        # Perform merges
        for i in range(num_merges):
            stats = self._get_stats(ids)
            if not stats:
                break
            
            pair = max(stats, key=stats.get)
            idx = 256 + i
            
            print(f"merging {pair} into a new token {idx}")
            
            ids = self._merge(ids, pair, idx)
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
        
        print(f"\nTraining complete!")
        print(f"Original length: {len(tokens)}")
        print(f"Compressed length: {len(ids)}")
        print(f"Compression ratio: {len(tokens) / len(ids):.2f}X")
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text into a list of token IDs.
        
        Args:
            text: Text to encode
            
        Returns:
            List of token IDs
        """
        if not self.merges:
            raise RuntimeError("Tokenizer has not been trained or loaded. Call train() or load() first.")
        
        # Start with UTF-8 bytes
        tokens = list(text.encode("utf-8"))
        
        # Apply merges
        while len(tokens) >= 2:
            stats = self._get_stats(tokens)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            
            if pair not in self.merges:
                break  # Nothing else can be merged
            
            idx = self.merges[pair]
            tokens = self._merge(tokens, pair, idx)
        
        return tokens
    
    def decode(self, ids: List[int]) -> str:
        """
        Decode a list of token IDs back into text.
        
        Args:
            ids: List of token IDs
            
        Returns:
            Decoded text string
        """
        if not self.vocab:
            raise RuntimeError("Tokenizer has not been trained or loaded. Call train() or load() first.")
        
        # Join all token bytes
        tokens = b"".join(self.vocab[idx] for idx in ids)
        
        # Decode UTF-8, replacing invalid sequences
        text = tokens.decode("utf-8", errors="replace")
        
        return text
    
    def save(self, vocab_path: str, merges_path: str) -> None:
        """
        Save vocabulary and merges to files.
        
        Args:
            vocab_path: Path to save vocabulary JSON
            merges_path: Path to save merges JSON
        """
        # Convert vocab bytes to base64 for JSON serialization
        vocab_serializable = {
            str(k): v.hex() for k, v in self.vocab.items()
        }
        
        # Convert tuple keys to string for JSON serialization
        merges_serializable = {
            f"{k[0]},{k[1]}": v for k, v in self.merges.items()
        }
        
        # Save vocabulary
        with open(vocab_path, 'w') as f:
            json.dump(vocab_serializable, f, indent=2)
        
        # Save merges
        with open(merges_path, 'w') as f:
            json.dump(merges_serializable, f, indent=2)
        
        print(f"Saved vocabulary to {vocab_path}")
        print(f"Saved merges to {merges_path}")
    
    def load(self, vocab_path: str, merges_path: str) -> None:
        """
        Load vocabulary and merges from files.
        
        Args:
            vocab_path: Path to vocabulary JSON file
            merges_path: Path to merges JSON file
        """
        # Load vocabulary
        with open(vocab_path, 'r') as f:
            vocab_serializable = json.load(f)
        
        # Convert hex strings back to bytes
        self.vocab = {
            int(k): bytes.fromhex(v) for k, v in vocab_serializable.items()
        }
        
        # Load merges
        with open(merges_path, 'r') as f:
            merges_serializable = json.load(f)
        
        # Convert string keys back to tuples
        self.merges = {
            tuple(map(int, k.split(','))): v for k, v in merges_serializable.items()
        }
        
        print(f"Loaded vocabulary from {vocab_path}")
        print(f"Loaded merges from {merges_path}")
    
    @staticmethod
    def _get_stats(ids: List[int]) -> Dict[Tuple[int, int], int]:
        """
        Count frequency of consecutive token pairs.
        
        Args:
            ids: List of token IDs
            
        Returns:
            Dictionary mapping pairs to their frequency
        """
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts
    
    @staticmethod
    def _merge(ids: List[int], pair: Tuple[int, int], idx: int) -> List[int]:
        """
        Merge all occurrences of a pair into a new token.
        
        Args:
            ids: List of token IDs
            pair: Pair of IDs to merge
            idx: New token ID to replace the pair
            
        Returns:
            New list with pairs merged
        """
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids
    
    def __repr__(self) -> str:
        return f"BPETokenizer(vocab_size={len(self.vocab)}, num_merges={len(self.merges)})"


if __name__ == "__main__":
    # Example usage
    sample_text = """Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 The very name strikes fear and awe into the hearts of programmers worldwide. We all know we ought to "support Unicode" in our software (whatever that means—like using wchar_t for all the strings, right?). But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus its dozens of supplementary annexes, reports, and notes can be more than a little intimidating. I don't blame programmers for still finding the whole thing mysterious, even 30 years after Unicode's inception."""
    
    # Create and train tokenizer
    tokenizer = BPETokenizer()
    tokenizer.train(sample_text, vocab_size=276)
    
    # Test encoding and decoding
    test_text = "hello world"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    
    print(f"\nTest encoding: '{test_text}'")
    print(f"Encoded: {encoded}")
    print(f"Decoded: '{decoded}'")
    print(f"Match: {test_text == decoded}")
    
    # Save to files
    tokenizer.save("vocab.json", "merges.json")
    
    # Test loading
    new_tokenizer = BPETokenizer()
    new_tokenizer.load("vocab.json", "merges.json")
    
    # Verify loaded tokenizer works
    encoded2 = new_tokenizer.encode(test_text)
    decoded2 = new_tokenizer.decode(encoded2)
    print(f"\nLoaded tokenizer test: {test_text == decoded2}")
