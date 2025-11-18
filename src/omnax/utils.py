import hashlib

import jax


def fold_in_str(key: jax.Array, string: str) -> jax.Array:
    """Returns a PRNG key derived from an initial PRNG key and a string input.

    Args:
      key: The initial PRNG key.
      string: The string input (e.g., 'pretrain', 'query', etc.).

    Returns:
      A PRNG key derived from the initial PRNG key and the string input.
    taken from https://github.com/MatX-inc/seqax/blob/main/jax_extra.py#L11
    """
    return jax.random.fold_in(key, int(hashlib.md5(string.encode()).hexdigest()[:8], base=16))
