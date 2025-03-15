import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from examples.text_onehot.model import ModelArgs, Transformer


UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"


def build_char_vocab(texts):
    chars = set([])
    for quote in tqdm(texts):
        text = quote.strip().strip('”“')
        chars.update(text)
    char_list = sorted(chars)
    char_list.append(UNK_TOKEN)
    char_list.append(PAD_TOKEN)
    vocab = {ch: id for id, ch in enumerate(char_list)}
    return vocab


class OneHotEmbedder:

    def __init__(self, vocab: dict, unk_token=UNK_TOKEN, pad_token=PAD_TOKEN):
        self.token2id = vocab
        self.id2token = {id: token for token, id in vocab.items()}
        self.unk_id = self.token2id[unk_token]
        self.pad_id = self.token2id[pad_token]

    def encode(self, texts: list[str]):
        max_len = max([len(text) for text in texts])
        # ids = [[[0] * max_len] for _ in texts]
        emb = np.zeros((len(texts), max_len, len(self.token2id)))
        for i, text in enumerate(texts):
            for j in range(max_len):
                if j < len(text):
                    ch = text[j]
                    k = self.token2id.get(ch, self.unk_id)
                else:
                    k = self.pad_id
                emb[i, j, k] = 1
        return jnp.array(emb)

    def decode(self, emb: jax.Array):
        assert emb.ndim == 3, "Expected array of shape (bsz, seq_len, dim)"
        ids = emb.argmax(axis=-1)
        texts = []
        for tids in ids:
            chars = [self.id2token[tid.item()] for tid in tids if tid.item() != self.pad_id]
            text = "".join(chars)
            texts.append(text)
        return texts


def main():
    vocab_size = 32
    args = ModelArgs(dim=vocab_size, n_layers=2, n_heads=8, vocab_size=vocab_size)
    model = Transformer(args)

    ds = load_dataset("Abirate/english_quotes")
    vocab = build_char_vocab(ds["train"]["quote"])
    texts = ["Let's have a black celebration", "Let's come together!"]
    self = OneHotEmbedder(vocab)
    emb = self.encode(texts)
    self.decode(emb)