import math
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
from collections import Counter


UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"


def closest_power_of_2(n: int, lower=True):
    log_n = math.log(n, 2)
    pow = math.floor(log_n) if lower else math.ceil(log_n)
    return 2**(pow)


def build_char_vocab(texts):
    chars = Counter()
    for quote in tqdm(texts):
        text = quote.strip().strip('”“')
        chars.update(text)
    target_vocab_size = closest_power_of_2(len(chars))
    char_list = [ch for ch, _ in chars.most_common(target_vocab_size - 2)]
    char_list.append(UNK_TOKEN)
    char_list.append(PAD_TOKEN)
    assert len(char_list) == target_vocab_size
    vocab = {ch: id for id, ch in enumerate(char_list)}
    return vocab


class OneHotEncoder:

    def __init__(self, vocab: dict, unk_token=UNK_TOKEN, pad_token=PAD_TOKEN, max_length: int = 512):
        self.token2id = vocab
        self.id2token = {id: token for token, id in vocab.items()}
        self.unk_id = self.token2id[unk_token]
        self.pad_id = self.token2id[pad_token]
        self.max_length = max_length

    def encode(self, texts: list[str]):
        length = max([len(text) for text in texts])
        length = closest_power_of_2(length, lower=False)
        length = min(length, self.max_length)
        bsz, length, vocab_size = len(texts), length, len(self.token2id)
        emb = np.zeros((bsz, length, vocab_size))
        pad_mask = np.ones((bsz, length), dtype=bool)
        for i, text in enumerate(texts):
            for j in range(length):
                if j < len(text):
                    ch = text[j]
                    k = self.token2id.get(ch, self.unk_id)
                else:
                    k = self.pad_id
                    pad_mask[i, j] = False
                emb[i, j, k] = 1
        return jnp.array(emb), jnp.array(pad_mask)

    def decode(self, emb: jax.Array):
        assert emb.ndim == 3, "Expected array of shape (bsz, seq_len, dim)"
        ids = emb.argmax(axis=-1)
        texts = []
        for tids in ids:
            chars = [self.id2token[tid.item()] for tid in tids if tid.item() != self.pad_id]
            text = "".join(chars)
            texts.append(text)
        return texts
