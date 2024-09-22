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
        pre_vocab = 256
        assert vocab_size > pre_vocab 
        encoded_text = text.encode("utf-8")
        tokens = list(map(int,encoded_text))

        
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
            stats = get_stats(tokens)
            pair = min(stats, key=lambda p: self.merges.get(p, float('inf')))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            tokens = merge(tokens, pair, idx)
        return tokens
    
    def decode(self, tokens):
        vocab = {idx : bytes([idx]) for idx in range(256)}
        for (p1,p2), idx in self.merges.items():
            vocab[idx] = vocab[p1] + vocab[p2]

        return b"".join(vocab[idx] for idx in tokens).decode("utf-8")
    
    def save(self, path :  str):
        o
        with open()


with open("taylorswift.txt", "r") as f:
    text = f.read()
t = BasicTokenizer()
t.train(text, vocab_size=300)

t.save()

print("""Copy paste of the Wikipedia article on Taylor Swift, as of Feb 16, 2024. oiwjeovhoweh 344 2""" ==
       t.decode(t.encode("""Copy paste of the Wikipedia article on Taylor Swift, as of Feb 16, 2024. oiwjeovhoweh 344 2""")))

print("""రూపొందుతోందని """ ==
       t.decode(t.encode("""రూపొందుతోందని """)))

print(t.encode("""రూపొందుతోందని """))