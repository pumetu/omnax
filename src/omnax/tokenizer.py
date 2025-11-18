import json
import os

from tokenizers import Tokenizer


class HuggingFaceTokenizer:
    def __init__(self, model_name_or_path: str):
        self.tokenizer = Tokenizer.from_file(os.path.join(model_name_or_path, "tokenizer.json"))
        self.config = json.load(open(os.path.join(model_name_or_path, "tokenizer_config.json")))

    @property
    def pad_token(self):
        return self.config["pad_token"]

    @property
    def pad_token_id(self):
        return self.tokenizer.token_to_id(self.pad_token)

    @property
    def eos_token(self):
        if "eos_token" in self.config:
            return self.config["eos_token"]
        else:
            return "<|endoftext|>"

    @property
    def eos_token_id(self):
        return self.tokenizer.token_to_id(self.eos_token)

    @property
    def vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def encode(self, text: str, prepend: str | int = None, append: str | int = None):
        ids = []
        if prepend is not None:
            preprend_id = prepend if isinstance(prepend, int) else self.tokenizer.token_to_id(prepend)
            ids.append(preprend_id)
        ids.extend(self.tokenizer.encode(text, add_special_tokens=False).ids)
        if append is not None:
            append_id = append if isinstance(append, int) else self.tokenizer.token_to_id(append)
            ids.append(append_id)
        return ids

    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=False)
