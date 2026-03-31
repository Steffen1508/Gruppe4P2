"""
Token-based chunking with customizable overlap (split on sentence boundaries)
"""

import re

def sentence_tokenize(text):
    """
    Split text into sentences based on period (.) punctuation.
    
    Parameters:
    -----------
    text : str
        The text to tokenize
    
    Returns:
    --------
    list[str]
        List of sentences (including the period)
    
    Example:
    --------
    >>> text = "Hello world. This is a test. Great!"
    >>> sentences = sentence_tokenize(text)
    >>> print(sentences)
    ['Hello world.', 'This is a test.', 'Great!']
    """
    # Split on periods while keeping them with the sentence
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    # Filter out empty strings
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def count_tokens(text):
    """
    Count approximate number of tokens in text (simple whitespace tokenization).
    
    Parameters:
    -----------
    text : str
        The text to count tokens for
    
    Returns:
    --------
    int
        Approximate number of tokens
    """
    return len(text.split())


def token_chunking(text, chunk_size, overlap=0, max_tokens=64):
    """
    Split text into fixed-size chunks of sentences with customizable overlap.
    Chunks are based on sentence boundaries (split after . ! ? punctuation).
    Each chunk respects a maximum token limit.
    
    Parameters:
    -----------
    text : str
        The text to chunk
    chunk_size : int
        Maximum number of sentences per chunk
    overlap : int
        Number of sentences to overlap between consecutive chunks (default: 0)
    max_tokens : int
        Maximum number of tokens allowed per chunk (default: 64)
    
    Returns:
    --------
    list[str]
        List of text chunks (each chunk contains multiple sentences up to max_tokens)
    
    Example:
    --------
    >>> text = "Hello world. This is a test. Great idea. Let's continue."
    >>> chunks = token_chunking(text, chunk_size=2, overlap=0, max_tokens=64)
    >>> for i, chunk in enumerate(chunks):
    ...     print(f"Chunk {i}: '{chunk}' ({count_tokens(chunk)} tokens)")
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size-1")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    
    # Split text into sentences
    sentences = sentence_tokenize(text)
    
    chunks = []
    step_size = chunk_size - overlap
    
    # Create chunks from sentences, respecting max_tokens limit
    i = 0
    while i < len(sentences):
        chunk_sentences = []
        token_count = 0
        j = i
        
        # Add sentences while respecting both chunk_size and max_tokens
        while j < len(sentences) and len(chunk_sentences) < chunk_size:
            sentence = sentences[j]
            sentence_tokens = count_tokens(sentence)
            
            # Check if adding this sentence would exceed max_tokens
            if token_count + sentence_tokens > max_tokens and chunk_sentences:
                # If we already have sentences, stop here
                break
            
            chunk_sentences.append(sentence)
            token_count += sentence_tokens
            j += 1
        
        if chunk_sentences:
            chunk = " ".join(chunk_sentences)
            chunks.append(chunk)
        
        i += step_size
    
    return chunks
def fixed_size_chunking_list(items, chunk_size, overlap=0):
    """
    Split a list of items into fixed-size chunks with customizable overlap.
    
    Parameters:
    -----------
    items : list
        List of items to chunk (can be tokens, sentences, etc.)
    chunk_size : int
        Size of each chunk (number of items)
    overlap : int
        Number of items to overlap between consecutive chunks (default: 0)
    
    Returns:
    --------
    list[list]
        List of chunks, where each chunk is a list of items
    
    Example:
    --------
    >>> items = list(range(10))
    >>> chunks = fixed_size_chunking_list(items, chunk_size=3, overlap=1)
    >>> for i, chunk in enumerate(chunks):
    ...     print(f"Chunk {i}: {chunk}")
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size-1")
    
    chunks = []
    step_size = chunk_size - overlap
    
    for i in range(0, len(items), step_size):
        chunk = items[i:i + chunk_size]
        chunks.append(chunk)
        
        if i + chunk_size >= len(items):
            break
    
    return chunks


# Example usage
if __name__ == "__main__":
    # Example 1: Sentence-based token chunking
    print("=" * 50)
    print("Example 1: Sentence-based Token Chunking")
    print("=" * 50)
    
    text = "The quick brown fox jumps over the lazy dog. This is a test of token-based chunking. It splits on sentence boundaries. This approach is better for NLP tasks. Let's see the results."
    
    print(f"\nOriginal text:\n{text}\n")
    
    # Extract sentences
    sentences = sentence_tokenize(text)
    print(f"Extracted sentences ({len(sentences)} total):")
    for i, sent in enumerate(sentences):
        print(f"  Sentence {i}: '{sent}'")
    
    # Without overlap
    print("\n\nChunking with chunk_size=2, overlap=0, max_tokens=64:")
    chunks = token_chunking(text, chunk_size=2, overlap=0, max_tokens=64)
    for i, chunk in enumerate(chunks):
        tokens = count_tokens(chunk)
        print(f"  Chunk {i}: ({tokens} tokens)\n    '{chunk}'")
    
    # With overlap
    print("\n\nChunking with chunk_size=2, overlap=1, max_tokens=64:")
    chunks = token_chunking(text, chunk_size=2, overlap=1, max_tokens=64)
    for i, chunk in enumerate(chunks):
        tokens = count_tokens(chunk)
        print(f"  Chunk {i}: ({tokens} tokens)\n    '{chunk}'")
    
    # With lower max_tokens
    print("\n\nChunking with chunk_size=3, overlap=0, max_tokens=30:")
    chunks = token_chunking(text, chunk_size=3, overlap=0, max_tokens=30)
    for i, chunk in enumerate(chunks):
        tokens = count_tokens(chunk)
        print(f"  Chunk {i}: ({tokens} tokens)\n    '{chunk}'")
    
    # Example 2: List chunking (useful for pre-tokenized data)
    print("\n" + "=" * 50)
    print("Example 2: List Chunking (Pre-tokenized Items)")
    print("=" * 50)
    
    tokens = ["word1", "word2", "word3", "word4", "word5", "word6", "word7", "word8"]
    
    print(f"\nOriginal tokens: {tokens}\n")
    
    print("Chunking with chunk_size=3, overlap=0:")
    chunks = fixed_size_chunking_list(tokens, chunk_size=3, overlap=0)
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i}: {chunk}")
    
    print("\nChunking with chunk_size=3, overlap=1:")
    chunks = fixed_size_chunking_list(tokens, chunk_size=3, overlap=1)
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i}: {chunk}")
    
    print("\nChunking with chunk_size=4, overlap=2:")
    chunks = fixed_size_chunking_list(tokens, chunk_size=4, overlap=2)
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i}: {chunk}")
