from dataclasses import dataclass, fields, is_dataclass
from functools import partial
from typing import Any, Callable

import jax
import jax.numpy as jnp


def pytree_dataclass(cls: Any, meta_fields: tuple[str] = ()):
    """
    Args:
        cls: the class to wrap
        meta_fields (tuple[str]): fields that will be treated as static fields
    """
    if not is_dataclass(cls):
        cls = dataclass(cls)
    all_fields = tuple(f.name for f in fields(cls) if f.init)
    data_fields = tuple(f for f in all_fields if f not in meta_fields)
    return jax.tree_util.register_dataclass(cls, data_fields=data_fields, meta_fields=meta_fields)


@partial(pytree_dataclass, meta_fields=("shape", "dtype", "initializer"))
class Spec:
    shape: tuple[int, ...]
    dtype: jnp.dtype
    initializer: Callable | None = None


is_type = lambda x, cls: (type(x).__name__ == cls.__name__) and (type(x).__module__ == cls.__module__)
is_param = lambda x: is_type(x, Spec)


class Module:
    @classmethod
    def spec(cls, *arg, **kwargs): ...

    @classmethod
    def init(cls, rng: jax.random.PRNGKey, **kwargs):
        def _init(rng, spec):
            num_params = len(jax.tree.leaves(spec, is_leaf=is_param))
            rngs = iter(jax.random.split(rng, num_params))
            return jax.tree.map(lambda param: param.initializer(next(rngs), param.shape, param.dtype), spec, is_leaf=is_param)

        spec = cls.spec(**kwargs)
        spec_leaves, spec_def = jax.tree.flatten(spec, is_leaf=is_param)
        return jax.tree.unflatten(spec_def, _init(rng, tuple(spec_leaves)))
