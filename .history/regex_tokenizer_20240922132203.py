from .base import BaseTokenizer, get_stats, merge


class RegexTokenizer(BaseTokenizer):

    def __init__(self) -> None:
        super().__init__()

    de