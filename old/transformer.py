# from transformers import AutoConfig

# from transformers import BertEncoder
# from transformers.models.bert.modeling_bert import BertEncoder
import math
import jax
import jax.numpy as jnp
import flax.nnx as nnx

from fabrique.models.bert.modeling import Transformer as Bert
from fabrique.loading import from_pretrained

# import torch as th
# import torch.nn as nn
# from src.modeling.diffusion.nn import (
#     SiLU,
#     linear,
#     timestep_embedding,
# )


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = jnp.exp(
        -math.log(max_period) * jnp.arange(start=0, stop=half, dtype=jnp.float32) / half
    )
    args = timesteps[:, None].astype(jnp.float32) * freqs[None]
    embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
    if dim % 2:
        embedding = jnp.concatenate([embedding, jnp.zeros_like(embedding[:, :1])], axis=-1)
    return embedding




def main():
    self = TransformerModel(768, 512, 768)
    x = self.tokenizer.encode_batch(["ping pong"])[0].ids
    x = jnp.array(x).reshape(1, -1)
    # todo: what is the shape of x? why lm_head is not used?



class TransformerModel(nnx.Module):
    """
    A transformer model to be used in Diffusion Model Training.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes. TODO for the next version
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        init_pretrained: bool = False,
        freeze_embeddings: bool = False,
        use_pretrained_embeddings: bool = True,
        dropout=0,
        use_checkpoint=False,
        num_heads=1,
        config=None,
        config_name="bert-base-uncased",
        vocab_size=None,
        logits_mode=1,
        rngs: nnx.Rngs = nnx.Rngs(0)
    ):
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.logits_mode = logits_mode
        self.vocab_size = vocab_size
        self.init_pretrained = init_pretrained
        self.freeze_embeddings = freeze_embeddings
        self.use_pretrained_embeddings = use_pretrained_embeddings
        self.config = config
        self.config_name = config_name
        self.rngs = rngs

        time_embed_dim = model_channels * 4

        self.tokenizer, self.temp_bert, self.config  = from_pretrained(self.config_name)
        import argparse
        self.config = argparse.Namespace(**self.config)
        del self.temp_bert.pooler
        self.input_transformers = self.temp_bert
        config = self.config

        self.time_embed = nnx.Sequential(
            nnx.Linear(model_channels, time_embed_dim, rngs=rngs),
            nnx.silu,
            nnx.Linear(time_embed_dim, config.hidden_size, rngs=rngs),
        )

        # self.build_xstart_predictor()
        self.build_input_output_projections()
        self.build_embeddings()

        # self.register_buffer(
        #     "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1))
        # )

        self.LayerNorm = nnx.LayerNorm(self.config.hidden_size, epsilon=self.config.layer_norm_eps, rngs=rngs)
        self.dropout = nnx.Dropout(self.config.hidden_dropout_prob, rngs=rngs)

    # def build_xstart_predictor(self):
    #     # self.tokenizer, self.temp_bert, self.config  = from_pretrained(self.config_name)
    #     del self.temp_bert.pooler
    #     self.input_transformers = self.temp_bert


    def build_input_output_projections(self):
        if self.use_pretrained_embeddings:
            self.input_up_proj = lambda x: x
            self.output_down_proj = lambda x: x
        else:  # need to adapt the model to the embedding size
            self.input_up_proj = nnx.Sequential(
                nnx.Linear(self.in_channels, self.config.hidden_size, rngs=self.rngs),
                jnp.tanh,
                nnx.Linear(self.config.hidden_size, self.config.hidden_size, rngs=self.rngs),
            )

            self.output_down_proj = nnx.Sequential(
                nnx.Linear(self.config.hidden_size, self.config.hidden_size, rngs=self.rngs),
                jnp.tanh,
                nnx.Linear(self.config.hidden_size, self.out_channels, rngs=self.rngs),
            )

    def build_embeddings(self):
        if self.use_pretrained_embeddings:
            self.word_embedding = self.temp_bert.embeddings.token_embeddings
            self.position_embeddings = self.temp_bert.embeddings.position_embeddings
        else:
            assert self.vocab_size is not None
            self.word_embedding = nnx.Embed(self.vocab_size, self.in_channels, rngs=self.rngs)
            self.position_embeddings = nnx.Embed(
                self.config.max_position_embeddings, self.config.hidden_size, rngs=self.rngs
            )

        self.lm_head = nnx.Linear(self.in_channels, self.word_embedding.embedding.value.shape[0], rngs=self.rngs)

        # if self.freeze_embeddings:
        #     self.word_embedding.weight.requires_grad = False
        #     self.position_embeddings.weight.requires_grad = False

        #with th.no_grad():
        self.lm_head.kernel.value = self.word_embedding.embedding.value.T  # note: params aren't bound

    def get_embeds(self, input_ids):
        return self.word_embedding(input_ids)

    def get_logits(self, hidden_repr):
        return self.lm_head(hidden_repr)

    def __call__(self, x, timesteps, y=None, attention_mask=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """

        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        emb_x = self.input_up_proj(x)
        seq_length = x.shape[1]
        position_ids = self.position_ids[:, :seq_length]
        emb_inputs = (
            self.position_embeddings(position_ids)
            + emb_x
            + emb.unsqueeze(1).expand(-1, seq_length, -1)
        )
        emb_inputs = self.dropout(self.LayerNorm(emb_inputs))

        # https://github.com/huggingface/transformers/blob/e95d433d77727a9babadf008dd621a2326d37303/src/transformers/modeling_utils.py#L700
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]

        input_trans_hidden_states = self.input_transformers(
            emb_inputs, attention_mask=attention_mask
        ).last_hidden_state

        h = self.output_down_proj(input_trans_hidden_states)
        h = h.type(x.dtype)
        return h