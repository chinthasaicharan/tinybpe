def get_stats(text, counts =None):
  counts= counts if counts is not None else counts
  for pair in zip(text,text[1:]):
    counts[pair] = counts.get(pair, 0) + 1
  return counts

def merge(ids, pair, idx):
  new_ids = []
  i = 0
  while (i<len(ids)):
    if  i < len(ids) -1 and  ids[i] == pair[0] and  ids[i+1] == pair[1]:
      new_ids.append(idx)
      i+=2
    else:
      new_ids.append(ids[i])
      i+=1
  return new_ids


class BaseTokenizer:
    def __init__(self) -> None:
        self.merges = {}
        self.special_tokens = {}
        self.vocab = self._build_vocab()

    def train(self, text : str, vocab_size : int):
        """
        Trains tokenizer on given
         text : str with vocab_size : int
        """

        assert vocab_size >= 256 
        encoded_text = text.encode("utf-8")
        tokens = list(map(int,encoded_text))

        
        num_merges = vocab_size - 256

        for i in range(num_merges):
            stats = get_stats(tokens)
            maxy = max(stats, key=stats.get)
            newidx = i + 256
            tokens = merge(tokens, maxy, newidx)
            self.merges[maxy] = newidx
        self.vocab = self._build_vocab()

    def _build_vocab(self):

        vocab = {idx : bytes([idx]) for idx in range(256)}
        for (p1,p2), idx in self.merges.items():
            vocab[idx] = vocab[p1] + vocab[p2]

        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

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
        return b"".join(self.vocab[idx] for idx in tokens).decode("utf-8")
    
    def save(self, file_name :  str):

        file = file_name + ".model"

        with open(file,"w", encoding="utf-8") as f:
            f.write(f"{len(self.special_tokens)}\n")
            for token, idx in self.special_tokens.items():
                f.write(f"{token} {idx}\n")

            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2} \n")

                

    def load(self, file_path : str):
        spl_tokens = {}
        merges= {}
        i = 256
        assert file_path.endswith(".model")
        with open(file_path, "r") as f:
            n_spltokens = int(f.readline().strip())
            for i in range(n_spltokens):
                token, idx = f.readline().strip().split(" ")
                spl_tokens[token] = token[idx]
            for line in f:
                idx1, idx2 = map(int, line.strip().split(" "))
                merges[(idx1,idx2)] = i
                i+=1

        self.merges = merges
        self.special_tokens = spl_tokens
        self.vocab = self._build_vocab()
