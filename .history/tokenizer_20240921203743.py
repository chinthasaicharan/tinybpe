from base import get_stats, merge
class BasicTokenizer:
    def __init__(self) -> None:
        pass

    def train(text : , vocab_size : int):
        "Trains tokenizer on given text : str with vocab_size : int"
        encoded_text = text.encode("utf-8", errors="replace")
        ids = list(map(int,encoded_text))

        vocab_size = 276
        pre_vocab = 256
        assert vocab_size > pre_vocab
        num_merges = vocab_size - pre_vocab
        tokens = list(ids)

        merges = {}
        for i in range(num_merges):
            stats = get_stats(tokens)
            maxy = max(stats, key=stats.get)
            newidx = i + 256
            tokens = merge(tokens, maxy, newidx)
            merges[maxy] = newidx