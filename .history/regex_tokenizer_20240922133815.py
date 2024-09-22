from .base import BaseTokenizer, get_stats, merge
import regex as re

# the main GPT text split patterns, see
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class RegexTokenizer(BaseTokenizer):
    """
    pattern : str -> regex pattern , default: gpt4 regex pattern
    """

    def __init__(self, pattern = None) -> None:
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {}
        self.inverse_special_tokens = {}

    def train(self, text: str, vocab_size: int):
        
        assert vocab_size >= 256 
        num_merges = vocab_size - 256

        text_chunks = re.findall(self.compiled_pattern, text) 
        int_tokens = [list(i.encode("utf-8")) for i in text_chunks]
        merges = {}
        vocab = {bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            stats = {}
            for word in text_chunks:
                get_stats(word, stats)

            max_pair = max(stats, key=stats.get)
            idx = i + 256
            int_tokens = [merge(word, max_pair, idx) for word in int_tokens]

            merges[max_pair] = idx
            vocab[idx] = vocab[max_pair[0]] + vocab[max_pair[1]]
            

        self.merges = merges
        self.vocab = vocab
