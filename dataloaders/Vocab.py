import collections

class Vocab(object):  #@save
    """文本词表
    tokens (list): 词元列表
    min_freq (int): 词元的最小频率，小于该频率的词元将被忽略
    reserved_tokens (list): 保留的词元列表，例如['<pad>', '<eos>']
    """
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=['<eos>', '<unk>', '<pad>']):
        if tokens is None:
            tokens = []

        if reserved_tokens is None:
            reserved_tokens = []

        # 按出现频率排序
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True) # 常见词在前，对cache友好
        # 未知词元的索引为0
        self.idx2token = reserved_tokens
        self.token2idx = {token: idx
                             for idx, token in enumerate(self.idx2token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token2idx:
                self.idx2token.append(token)
                self.token2idx[token] = len(self.idx2token) - 1

    def __len__(self):
        return len(self.idx2token)

    def __getitem__(self, tokens):
        """
        Args:
            tokens (str or list or tuple): 词元或词元列表
        Returns:
            int or list or tuple: 词元索引或词元索引列表
        """
        if not isinstance(tokens, (list, tuple)):
            return self.token2idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx2token[indices]
        return [self.idx2token[index] for index in indices]

    @property
    def unk(self):  # 未知词元的索引为0
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs

def count_corpus(tokens):  #@save
    """统计词元的频率"""
    # 这里的tokens是1D列表或2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成一个列表
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)