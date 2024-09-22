from .base import BaseTokenizer, get_stats, merge
import regex as re

class RegexTokenizer(BaseTokenizer):

    def __init__(self) -> None:
        super().__init__()

    def train(self, text: str, vocab_size: int):
        return super().train(text, vocab_size)