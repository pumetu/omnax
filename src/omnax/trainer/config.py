from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass
class DataConfig:
    dataset_name: str


@dataclass
class ModelConfig:
    vocab_size: int = 151936
    sequence_len: int = 256
    hidden_size: int = 256
    dtype: jax.typing.DTypeLike = jnp.float32
    num_hidden_layers: int = 2
    num_attention_heads: int = 1
    bias: bool = False


@dataclass
class Config:
    model_name_or_path: str
    data: DataConfig
    model: ModelConfig
    seed: int = 42
    dtype: jax.typing.DTypeLike = jnp.float32
    learning_rate: float = 2e-5
    batch_size: int = 2
