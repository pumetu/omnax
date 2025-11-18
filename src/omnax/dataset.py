from collections import deque

import jax.numpy as jnp
import numpy as np

from datasets import Dataset
from omnax.tokenizer import HuggingFaceTokenizer


def build_dataloader(batch_size: int, sequence_len: int, tokenizer: HuggingFaceTokenizer, dataset: Dataset, tokenizer_batch_size: int = 32):
    num_tokens = batch_size * sequence_len + 1
    token_buffer = deque()
    iter_dataset = iter(dataset.to_iterable_dataset())

    scratch = np.empty(num_tokens, dtype=np.int64)

    while True:
        while len(token_buffer) < num_tokens:
            example = next(iter_dataset)
            tokens = tokenizer.encode(example["text"], append=tokenizer.eos_token)
            token_buffer.extend(tokens)
        for i in range(num_tokens):
            scratch[i] = token_buffer.popleft()
        inputs = jnp.array(scratch[:-1], dtype=jnp.int32).reshape((batch_size, sequence_len))
        targets = jnp.array(scratch[1:], dtype=jnp.int32).reshape((batch_size, sequence_len))
        yield inputs, targets
