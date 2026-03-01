import tiktoken
from typing import List, Dict, Any
from config.settings import settings # Corrected import path
from nltk.tokenize import sent_tokenize

# Ensure NLTK data is available (run this once if not already downloaded)
# try:
#     sent_tokenize("test")
# except LookupError:
#     import nltk
#     nltk.download('punkt')

class ChunkOptimizer:
    def __init__(self, tokenizer_model_name: str = "gpt-4"):
        """
        Initializes the ChunkOptimizer with a tokenizer.
        Args:
            tokenizer_model_name (str): The name of the model to get the tokenizer for (e.g., 'gpt-4').
                                        Defaults to 'gpt-4' but can be overridden by EmbeddingManager's tokenizer.
        """
        self.tokenizer = tiktoken.encoding_for_model(tokenizer_model_name)

    def set_tokenizer(self, tokenizer: Any):
        """
        Sets the tokenizer to be used, typically from an EmbeddingManager.
        Args:
            tokenizer: The tokenizer instance from the embedding model.
        """
        self.tokenizer = tokenizer

    def chunk_text(self, text: str) -> List[str]:
        """
        Segments raw text into smaller, meaningful chunks based on sentences.

        Args:
            text (str): The raw input text.

        Returns:
            List[str]: A list of text chunks.
        """
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        for sentence in sentences:
            current_chunk.append(sentence)
            if len(current_chunk) >= settings.CHUNK_SIZE_SENTENCES:
                chunks.append(" ".join(current_chunk))
                # Implement overlap by keeping the last few sentences
                current_chunk = current_chunk[-settings.CHUNK_OVERLAP_SENTENCES:]
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks

    def count_tokens(self, text: str) -> int:
        """
        Uses the tokenizer to accurately count tokens for a given text.

        Args:
            text (str): The input text.

        Returns:
            int: The number of tokens in the text.
        """
        return len(self.tokenizer.encode(text))

    def optimize_chunks_for_context(self, chunks: List[Dict[str, Any]], query_tokens: int) -> List[Dict[str, Any]]:
        """
        Selects and potentially truncates chunks to fit within the maximum context window.
        Assumes chunks come with a 'score' key (e.g., combined similarity, recency, importance).
        Includes a conceptual placeholder for rudimentary synthesis/summarization.

        Args:
            chunks (List[Dict[str, Any]]): A list of retrieved memory chunks,
                                          each expected to have 'content' and 'score'.
            query_tokens (int): The number of tokens already consumed by the query or prompt itself.

        Returns:
            List[Dict[str, Any]]: An optimized list of chunks that fit within the token limit,
                                  potentially truncated or synthesized.
        """
        if not chunks: # Handle empty chunks list
            return []

        # Sort chunks by score in descending order
        sorted_chunks = sorted(chunks, key=lambda x: x.get('score', 0), reverse=True)

        optimized_selected_chunks = []
        current_tokens = query_tokens

        for chunk in sorted_chunks:
            chunk_content = chunk.get('content', '')
            chunk_tokens = self.count_tokens(chunk_content)

            if current_tokens + chunk_tokens <= settings.MAX_CONTEXT_TOKENS:
                optimized_selected_chunks.append(chunk)
                current_tokens += chunk_tokens
            else:
                # Conceptual Placeholder: Dynamic Synthesis/Summarization
                # If many relevant chunks are competing for space, or a long chunk needs to be fit,
                # a small LLM or heuristic could synthesize a shorter version.
                # For now, we continue with truncation.
                # if _can_be_synthesized(chunk_content, remaining_tokens):
                #     synthesized_content = _perform_synthesis(chunk_content, remaining_tokens)
                #     ... add synthesized_content ...

                # Attempt truncation if adding the full chunk exceeds limit
                remaining_tokens = settings.MAX_CONTEXT_TOKENS - current_tokens
                if remaining_tokens > 0:
                    truncated_content = self._truncate_text_by_tokens(chunk_content, remaining_tokens)
                    if truncated_content: # Only add if truncation resulted in some content
                        truncated_chunk = chunk.copy()
                        truncated_chunk['content'] = truncated_content
                        optimized_selected_chunks.append(truncated_chunk)
                        current_tokens += self.count_tokens(truncated_content)
                break # No more room for further chunks

        return optimized_selected_chunks

    def _truncate_text_by_tokens(self, text: str, max_tokens: int) -> str:
        """
        Truncates text to a maximum number of tokens, trying to respect sentence boundaries.
        """
        if max_tokens <= 0: # Ensure max_tokens is positive
            return ""

        tokens = self.tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text

        # Decode tokens up to max_tokens
        truncated_tokens = tokens[:max_tokens]
        truncated_text = self.tokenizer.decode(truncated_tokens)

        # Attempt to end at a natural sentence boundary if possible
        sentences = sent_tokenize(truncated_text)
        if sentences:
            # Try to include as many full sentences as possible without exceeding max_tokens
            final_text_parts = []
            current_token_count = 0
            for sentence in sentences:
                sentence_tokens = self.count_tokens(sentence)
                if current_token_count + sentence_tokens <= max_tokens:
                    final_text_parts.append(sentence)
                    current_token_count += sentence_tokens
                else:
                    break
            if final_text_parts:
                return " ".join(final_text_parts)
        
        return truncated_text.rsplit(' ', 1)[0] + "..." if ' ' in truncated_text else truncated_text # Fallback to word split if no sentence boundary

# Example usage (for testing purposes)
if __name__ == "__main__":
    # Make sure NLTK punkt tokenizer data is downloaded for sent_tokenize
    # import nltk
    # nltk.download('punkt')

    # Dummy chunks for testing optimize_chunks_for_context
    test_chunks = [
        {'id': 'c1', 'content': 'This is a very important chunk of information. It talks about many things. The first thing is crucial.', 'score': 0.9},
        {'id': 'c2', 'content': 'This is another chunk, less important than the first. It provides some background details.', 'score': 0.7},
        {'id': 'c3', 'content': 'A third piece of data, quite long and less relevant. It contains a lot of filler words and extra context that might not be strictly necessary.', 'score': 0.5},
        {'id': 'c4', 'content': 'A final, moderately important chunk.', 'score': 0.8},
    ]

    # Initialize ChunkOptimizer (using default gpt-4 tokenizer for now)
    chunk_optimizer = ChunkOptimizer()

    # Test chunk_text
    long_text = "This is the first sentence. This is the second sentence. This is the third sentence. And here is the fourth one. Finally, the fifth sentence. A sixth sentence to test overlap."
    print(f"Original text: {long_text}")
    text_chunks = chunk_optimizer.chunk_text(long_text)
    print(f"Chunked text ({len(text_chunks)} chunks):")
    for i, chunk in enumerate(text_chunks):
        print(f"  Chunk {i+1} ({chunk_optimizer.count_tokens(chunk)} tokens): {chunk}")

    # Test count_tokens
    sample_text = "Hello, world! How are you doing today?"
    tokens_count = chunk_optimizer.count_tokens(sample_text)
    print(f"'{sample_text}' has {tokens_count} tokens.")

    # Test optimize_chunks_for_context with a mock MAX_CONTEXT_TOKENS (e.g., 50 tokens)
    # Simulate settings.MAX_CONTEXT_TOKENS and query_tokens
    original_max_tokens = settings.MAX_CONTEXT_TOKENS
    settings.MAX_CONTEXT_TOKENS = 50 # Temporarily set a small limit for testing
    query_tokens_mock = 10

    print(f"Optimizing chunks for context (MAX_CONTEXT_TOKENS={settings.MAX_CONTEXT_TOKENS}, query_tokens={query_tokens_mock})...")
    optimized_chunks = chunk_optimizer.optimize_chunks_for_context(test_chunks, query_tokens_mock)

    total_optimized_tokens = query_tokens_mock + sum(self.count_tokens(c['content']) for c in optimized_chunks)
    print(f"Total tokens after optimization: {total_optimized_tokens}")

    for i, chunk in enumerate(optimized_chunks):
        print(f"  Optimized Chunk {i+1} (Score: {chunk.get('score', 'N/A')}, Tokens: {self.count_tokens(chunk['content'])}): {chunk['content']}")
    
    # Restore original setting
    settings.MAX_CONTEXT_TOKENS = original_max_tokens
