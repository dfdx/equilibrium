import jax.numpy as jnp

from fabrique.loading import IGNORE
from fabrique.loading import ConversionRule as R

# fmt: off
RULES = [
    R("embeddings.LayerNorm.weight", "embeddings.norm.scale"),
    R("embeddings.LayerNorm.bias", "embeddings.norm.bias"),
    R("embeddings.position_ids", IGNORE),  # TODO: are these learned positional embeddings?
    R("embeddings.position_embeddings.weight", "embeddings.position_embeddings.embedding"),
    R("embeddings.word_embeddings.weight", "embeddings.token_embeddings.embedding"),
    R("embeddings.token_type_embeddings.weight", "embeddings.segment_embeddings.embedding"),

    R("encoder.layer.{n}.attention.output.LayerNorm.weight", "layers[{n}].attention.norm.scale"),
    R("encoder.layer.{n}.attention.output.LayerNorm.bias", "layers[{n}].attention.norm.bias"),
    R("bert.encoder.layer.{n}.attention.self.query.weight", "layers[{n}].attention.wq.kernel", jnp.transpose),
    R("bert.encoder.layer.{n}.attention.self.query.bias", "layers[{n}].attention.wq.bias"),
    R("bert.encoder.layer.{n}.attention.self.key.weight", "layers[{n}].attention.wk.kernel", jnp.transpose),
    R("bert.encoder.layer.{n}.attention.self.key.bias", "layers[{n}].attention.wk.bias"),
    R("bert.encoder.layer.{n}.attention.self.value.weight", "layers[{n}].attention.wv.kernel", jnp.transpose),
    R("bert.encoder.layer.{n}.attention.self.value.bias", "layers[{n}].attention.wv.bias"),
    R("bert.encoder.layer.{n}.attention.output.dense.weight", "layers[{n}].attention.wo.kernel", jnp.transpose),
    R("bert.encoder.layer.{n}.attention.output.dense.bias", "layers[{n}].attention.wo.bias"),

    R("bert.encoder.layer.{n}.intermediate.dense.weight", "layers[{n}].feed_forward.w1.kernel", jnp.transpose),
    R("bert.encoder.layer.{n}.intermediate.dense.bias", "layers[{n}].feed_forward.w1.bias"),
    R("bert.encoder.layer.{n}.output.dense.weight", "layers[{n}].feed_forward.w2.kernel", jnp.transpose),
    R("bert.encoder.layer.{n}.output.dense.bias", "layers[{n}].feed_forward.w2.bias"),
    R("bert.encoder.layer.{n}.output.LayerNorm.gamma", "layers[{n}].feed_forward.norm.scale"),
    R("bert.encoder.layer.{n}.output.LayerNorm.beta", "layers[{n}].feed_forward.norm.bias"),

    R("bert.pooler.dense.weight", "pooler.w.kernel", jnp.transpose),
    R("bert.pooler.dense.bias", "pooler.w.bias"),

    R("cls.*", IGNORE),
]
# fmt: on