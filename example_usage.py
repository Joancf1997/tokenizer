"""
Example usage of the BPE Tokenizer

This script demonstrates how to use the BPETokenizer class for
training, encoding, decoding, and persisting tokenizers.
"""

from bpe_tokenizer import BPETokenizer


def example_train_and_save():
    """Example: Train a tokenizer and save it"""
    print("=" * 50)
    print("Example 1: Training and Saving")
    print("=" * 50)
    
    # Sample training text
    training_text = """
    Machine learning is fascinating! It enables computers to learn from data.
    Deep learning, a subset of machine learning, uses neural networks with multiple layers.
    Natural language processing helps computers understand human language.
    """
    
    # Create and train tokenizer
    tokenizer = BPETokenizer()
    tokenizer.train(training_text, vocab_size=280)
    
    # Save to files
    tokenizer.save("my_vocab.json", "my_merges.json")
    print("\n✅ Tokenizer saved successfully!\n")


def example_load_and_use():
    """Example: Load a pre-trained tokenizer and use it"""
    print("=" * 50)
    print("Example 2: Loading and Using")
    print("=" * 50)
    
    # Load the pre-trained tokenizer
    tokenizer = BPETokenizer()
    tokenizer.load("vocab.json", "merges.json")
    
    # Test with various texts
    test_texts = [
        "Hello, world!",
        "Machine learning is amazing!",
        "Unicode: 안녕하세요 👋",
    ]
    
    print("\nEncoding and decoding test:")
    print("-" * 50)
    for text in test_texts:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        
        print(f"Original:  {text}")
        print(f"Encoded:   {encoded}")
        print(f"Decoded:   {decoded}")
        print(f"Match:     {text == decoded}")
        print()


def example_compare_compression():
    """Example: Compare compression ratios for different texts"""
    print("=" * 50)
    print("Example 3: Compression Analysis")
    print("=" * 50)
    
    # Load tokenizer
    tokenizer = BPETokenizer()
    tokenizer.load("vocab.json", "merges.json")
    
    # Different types of text
    texts = {
        "Short text": "Hello",
        "Repeated words": "hello hello hello world world world",
        "Technical text": "The Unicode standard defines encoding schemes",
    }
    
    print("\nCompression ratios:")
    print("-" * 50)
    for name, text in texts.items():
        utf8_bytes = len(text.encode("utf-8"))
        encoded = tokenizer.encode(text)
        tokens = len(encoded)
        ratio = utf8_bytes / tokens if tokens > 0 else 0
        
        print(f"{name}:")
        print(f"  UTF-8 bytes: {utf8_bytes}")
        print(f"  Tokens:      {tokens}")
        print(f"  Ratio:       {ratio:.2f}x")
        print()


def main():
    """Run all examples"""
    try:
        example_train_and_save()
        example_load_and_use()
        example_compare_compression()
        
        print("=" * 50)
        print("All examples completed successfully! ✨")
        print("=" * 50)
        
    except FileNotFoundError:
        print("⚠️  Error: vocab.json or merges.json not found")
        print("Run 'python bpe_tokenizer.py' first to generate these files")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
