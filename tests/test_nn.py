from dataclasses import dataclass

import jax
import numpy as np
from flax import nnx

import omnax.nn as nn


def test_linear():
    def _test_linear(x, in_dim, out_dim):
        # create in omnax
        rng = jax.random.PRNGKey(42)
        model = nn.Linear.init(rng=rng, in_dim=in_dim, out_dim=out_dim, bias=True)
        out = model(x)

        # create in flax
        flax_model = nnx.Linear(in_dim, out_dim, rngs=nnx.Rngs(42))
        flax_model.kernel.value = model.weight
        flax_model.bias.value = model.bias
        flax_out = flax_model(x)

        # jax.test_util.check_grads(out, flax_out)
        np.testing.assert_allclose(np.array(out), np.array(flax_out))

    batch, sequence_len, in_dim, out_dim = 4, 2, 8, 16

    rng = jax.random.PRNGKey(100)
    _, test1_rng, test2_rng = jax.random.split(rng, num=3)
    _test_linear(jax.random.normal(test1_rng, shape=(batch, in_dim)), in_dim, out_dim)
