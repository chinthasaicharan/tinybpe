from base import get_stats, merge
class BasicTokenizer:
    def __init__(self) -> None:
        self.merges = {}
        self.ids = []

    def train(self, text : str, vocab_size : int):
        """
        Trains tokenizer on given
         text : str with vocab_size : int
        """

        assert vocab_size > pre_vocab 
        encoded_text = text.encode("utf-8")
        tokens = list(map(int,encoded_text))

        pre_vocab = 256
        num_merges = vocab_size - pre_vocab

        for i in range(num_merges):
            stats = get_stats(tokens)
            maxy = max(stats, key=stats.get)
            newidx = i + 256
            tokens = merge(tokens, maxy, newidx)
            self.merges[maxy] = newidx

    def encode(self, text : str):
        tokens = list(text.encode("utf-8"))
        while True:
            