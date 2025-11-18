from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax
import tyro

from datasets import load_dataset
from omnax import pytree_dataclass
from omnax.dataset import build_dataloader
from omnax.models import Transformer
from omnax.tokenizer import HuggingFaceTokenizer
from omnax.trainer.config import Config
from omnax.utils import fold_in_str


@dataclass
class PreTrainConfig(Config):
    steps: int = 10000


@pytree_dataclass
class State:
    model: Transformer
    opt_state: optax.OptState

    @staticmethod
    def init(rng: jax.random.PRNGKey, config: PreTrainConfig, optimizer: optax.GradientTransformation):
        model = Transformer.init(rng, config=config)
        return State(model=model, opt_state=optimizer.init(model))


def main():
    config = tyro.cli(PreTrainConfig)
    tokenizer = HuggingFaceTokenizer(config.model_name_or_path)
    rng = jax.random.PRNGKey(config.seed)

    dataset = load_dataset(config.data.dataset_name, split="train")
    dataloader = build_dataloader(config.batch_size, config.model.sequence_len, tokenizer, dataset)
    x, y = next(dataloader)  # kick off load first batch

    optimizer = optax.adam(learning_rate=config.learning_rate)
    state = State.init(rng, config.model, optimizer)

    @jax.jit
    def train_step(state: State, x, y) -> tuple[State, float]:
        loss, grads = jax.value_and_grad(lambda model: model(x, y))(state.model)
        updates, opt_state = optimizer.update(grads, state.opt_state, state.model)
        state = State(model=optax.apply_updates(state.model, updates), opt_state=opt_state)
        return state, loss

    for i in range(config.steps):
        state, loss = train_step(state, x, y)
        x, y = next(dataloader)

        if i % 1000 == 0:
            print(loss)

    inputs = jnp.expand_dims(jnp.array(tokenizer.encode("One day")), axis=0)
    output = state.model.generate(fold_in_str(rng, "generate"), inputs)
    print(tokenizer.decode(output[0].tolist()))


if __name__ == "__main__":
    main()
