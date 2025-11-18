from functools import partial

import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, Int

import omnax.nn as nn
from omnax import Module, Spec, pytree_dataclass
from omnax.trainer.config import ModelConfig


@pytree_dataclass
class Attention(Module):
    wq: nn.Einsum
    wk: nn.Einsum
    wv: nn.Einsum
    wo: nn.Einsum

    @classmethod
    def spec(cls, hidden_size: int, dtype: jnp.dtype, bias: bool = False):
        return cls(
            wq=nn.Einsum.spec(subscripts="bsd,dd->bsd", hidden_size=hidden_size, dtype=dtype, bias=bias),
            wk=nn.Einsum.spec(subscripts="bsd,dd->bsd", hidden_size=hidden_size, dtype=dtype, bias=bias),
            wv=nn.Einsum.spec(subscripts="bsd,dd->bsd", hidden_size=hidden_size, dtype=dtype, bias=bias),
            wo=nn.Einsum.spec(subscripts="bsd,dd->bsd", hidden_size=hidden_size, dtype=dtype, bias=bias),
        )

    def __call__(
        self, x: Float[Array, "batch sequence_len hidden_size"]
    ) -> Float[Array, "batch sequence_len hidden_size"]:
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        attn = jnp.einsum("bqd,bkd->bqk", q, k) / jnp.sqrt(k.shape[-1])
        attn = jnp.where(jnp.triu(jnp.ones_like(attn, dtype=bool), k=1), -jnp.inf, attn)
        attn = jax.nn.softmax(attn)
        out = jnp.einsum("bss,bsd->bsd", attn, v)
        return self.wo(out)


@pytree_dataclass
class MLP(Module):
    down_proj: nn.Linear
    gate_proj: nn.Linear
    up_proj: nn.Linear

    @classmethod
    def spec(cls, hidden_size: int, dtype: jnp.dtype, bias: bool = False):
        return cls(
            down_proj=nn.Linear.spec(in_dim=hidden_size * 3, out_dim=hidden_size, dtype=dtype, bias=bias),
            gate_proj=nn.Linear.spec(in_dim=hidden_size, out_dim=hidden_size * 3, dtype=dtype, bias=bias),
            up_proj=nn.Linear.spec(in_dim=hidden_size, out_dim=hidden_size * 3, dtype=dtype, bias=bias),
        )

    def __call__(
        self, x: Float[Array, "batch sequence_len hidden_size"]
    ) -> Float[Array, "batch sequence_len hidden_size"]:
        return self.down_proj(jax.nn.silu(self.gate_proj(x)) * self.up_proj(x))


@pytree_dataclass
class Block(Module):
    attn: Attention
    ffwd: MLP
    norm1: nn.LayerNorm
    norm2: nn.LayerNorm

    @classmethod
    def spec(cls, hidden_size: int, dtype: jnp.dtype, bias: bool = True):
        return cls(
            attn=Attention.spec(hidden_size=hidden_size, dtype=dtype, bias=bias),
            ffwd=MLP.spec(hidden_size=hidden_size, dtype=dtype, bias=bias),
            norm1=nn.LayerNorm.spec(hidden_size=hidden_size, bias=bias),  # TODO: switch to rmsnorm
            norm2=nn.LayerNorm.spec(hidden_size=hidden_size, bias=bias),
        )

    def __call__(
        self, x: Float[Array, "batch sequence_len hidden_size"]
    ) -> Float[Array, "batch sequence_len hidden_size"]:
        x = x + self.attn(self.norm1(x))
        return x + self.ffwd(self.norm2(x))


@partial(pytree_dataclass, meta_fields=("config",))
class Transformer(Module):
    config: ModelConfig
    tok_embeds: Spec | Array
    pos_embeds: Spec | Array
    layers: list[Block]
    norm: nn.LayerNorm
    lm_head: nn.Linear

    @classmethod
    def spec(cls, config: ModelConfig):
        # https://github.com/google-research/big_vision/blob/main/big_vision/models/ppp/gemma.py
        _init = jax.nn.initializers.variance_scaling(scale=1.0, mode="fan_in", distribution="normal")
        return cls(
            config=config,
            tok_embeds=Spec(shape=(config.vocab_size, config.hidden_size), dtype=config.dtype, initializer=_init),
            pos_embeds=Spec(shape=(config.sequence_len, config.hidden_size), dtype=config.dtype, initializer=_init),
            layers=[
                Block.spec(hidden_size=config.hidden_size, dtype=config.dtype, bias=config.bias)
                for _ in range(config.num_hidden_layers)
            ],
            norm=nn.LayerNorm.spec(hidden_size=config.hidden_size, bias=config.bias),
            lm_head=nn.Linear.spec(in_dim=config.hidden_size, out_dim=config.vocab_size, bias=False),
        )

    def __call__(
        self,
        x: Int[Array, "batch sequence_len"],
        targets: Int[Array, "batch sequence_len"] = None,
    ) -> Float[Array, "batch sequence_len vocab_size"] | float:
        x = self.tok_embeds[x] + self.pos_embeds[jnp.arange(x.shape[-1])]
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.lm_head(x)

        if targets is not None:
            logits = logits.astype(jnp.float32)
            targets = jax.nn.one_hot(targets, num_classes=self.config.vocab_size)
            loss = optax.softmax_cross_entropy(logits=logits, labels=targets)
            return loss.mean()
        return logits

    @partial(jax.jit, static_argnames=["temperature"])
    def sample(self, rng: jax.random.PRNGKey, x: Int[Array, "batch sequence_len"], temperature: float = 0.2):
        """Sample a single next token prediction given context (batch, sequence_length). Returns (batch , 1)"""
        assert temperature >= 0.0, "temperature must be non-negative"
        logits = self(x)[:, -1, :] / temperature
        return jnp.expand_dims(jax.random.categorical(rng, logits, axis=-1), axis=0)

    def generate(self, rng: jax.random.PRNGKey, x: Int[Array, "batch sequence_len"]):
        """Generate"""
        counter = 0
        while True:
            rng, sampling_rng = jax.random.split(rng)
            output = self.sample(sampling_rng, x)
            x = jnp.concatenate([x, output], axis=1)
            counter += 1
            if counter > 100:
                break
        return x
