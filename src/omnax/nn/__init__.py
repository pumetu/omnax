from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float32

from omnax import Module, Spec, pytree_dataclass


@pytree_dataclass
class Linear(Module):
    weight: Spec | Array
    bias: Spec | Array | None

    @classmethod
    def spec(cls, in_dim: int, out_dim: int, bias: bool = False, dtype: jnp.dtype = jnp.float32):
        _init = jax.nn.initializers.xavier_uniform()
        return cls(
            weight=Spec(shape=(in_dim, out_dim), dtype=dtype, initializer=_init),
            bias=Spec(shape=(out_dim,), dtype=dtype, initializer=_init) if bias else None,
        )

    def __call__(self, x: Array) -> Array:
        x = x @ self.weight
        return x + self.bias if self.bias is not None else x


@partial(pytree_dataclass, meta_fields=("subscripts",))
class Einsum(Module):
    subscripts: str
    weight: Spec | Array
    bias: Spec | Array | None

    @classmethod
    def spec(cls, subscripts: str, hidden_size: int, bias: bool = False, dtype: jnp.dtype = jnp.float32):
        _init = jax.nn.initializers.xavier_uniform()
        return cls(
            subscripts=subscripts,
            weight=Spec(shape=(hidden_size, hidden_size), dtype=dtype, initializer=_init),
            bias=Spec(shape=(hidden_size,), dtype=dtype, initializer=_init) if bias else None,
        )

    def __call__(self, x: Array) -> Array:
        x = jnp.einsum(self.subscripts, x, self.weight)
        return x + self.bias if self.bias is not None else x


@partial(pytree_dataclass, meta_fields=("eps",))
class LayerNorm(Module):
    weight: Spec | Array
    bias: Spec | Array | None
    eps: float

    @classmethod
    def spec(cls, hidden_size: int, bias: bool = True, eps: float = 1e-5):
        return cls(
            weight=Spec(shape=(hidden_size,), dtype=jnp.float32, initializer=jax.nn.initializers.ones),
            bias=Spec(shape=(hidden_size,), dtype=jnp.float32, initializer=jax.nn.initializers.zeros),
            eps=eps,
        )

    def __call__(self, x) -> Float32[Array, "..."]:
        mean, var = jnp.mean(x, axis=-1, keepdims=True), jnp.var(x, axis=-1, keepdims=True)
        x = (x - mean) / (jnp.sqrt(var) + self.eps) * self.weight
        return x + self.bias if self.bias is not None else x
