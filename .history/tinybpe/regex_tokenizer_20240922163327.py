from tinybpe.base import BaseTokenizer, get_stats, merge
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

    def train(self, text: str, vocab_size: int):
        assert vocab_size >= 256 
        num_merges = vocab_size - 256

        text_chunks = re.findall(self.compiled_pattern, text) 
        int_tokens = [list(i.encode("utf-8")) for i in text_chunks]
        merges = {}
        vocab = {idx : bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            stats = {}
            for word in int_tokens:
                get_stats(word, stats)

            max_pair = max(stats, key=stats.get)
            idx = i + 256
            int_tokens = [merge(word, max_pair, idx) for word in int_tokens]

            merges[max_pair] = idx
            vocab[idx] = vocab[max_pair[0]] + vocab[max_pair[1]]

        self.merges = merges
        self.vocab = vocab
    
    def _encode_chunk(self, text_bytes):
        # return the token ids
        # let's begin. first, convert all bytes to integers in range 0..255
        ids = list(text_bytes)
        while len(ids) >= 2:
            # find the pair with the lowest merge index
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # subtle: if there are no more merges available, the key will
            # result in an inf for every single pair, and the min will be
            # just the first pair in the list, arbitrarily
            # we can detect this terminating case by a membership check
            if pair not in self.merges:
                break # nothing else can be merged anymore
            # otherwise let's merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids
    
    def encode(self, text):
        """Encoding that ignores any special tokens."""
        # split text into chunks of text by categories defined in regex pattern
        text_chunks = re.findall(self.compiled_pattern, text)
        # all chunks of text are encoded separately, then results are joined
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8") # raw bytes
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids

    def decode(self, ids):
    # given ids (list of integers), return Python string
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            else:
                raise ValueError(f"invalid token id: {idx}")
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text


with open("taylorswift.txt", "r") as f:
    text = f.read()

t = RegexTokenizer()
t.train(text, vocab_size=300)
# t.save("taytoken")
# t.load("taytoken.model")


# print("""Copy paste of the Wikipedia article on Taylor Swift, as of Feb 16, 2024. oiwjeovhoweh 344 2""" ==
#        t.decode(t.encode("""Copy paste of the Wikipedia article on Taylor Swift, as of Feb 16, 2024. oiwjeovhoweh 344 2""")))

print("""రూపొందుతోందని """ ==
       t.decode(t.encode("""రూపొందుతోందని """)))

# print(t.encode("""రూపొందుతోందని """))