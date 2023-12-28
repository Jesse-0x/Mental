import mlx.core as mx
from typing import List, Optional, Tuple
from mlx.utils import tree_flatten, tree_map, tree_unflatten
from sentencepiece import SentencePieceProcessor
from models import LoRALinear, Model, ModelArgs
from pathlib import Path
import json

import cProfile

class Tokenizer:
    def __init__(self, model_path: str):
        assert Path(model_path).exists(), model_path
        self._model = SentencePieceProcessor(model_file=model_path)
        self._sep = "▁"
        assert self._model.vocab_size() == self._model.get_piece_size()

    def encode(self, s: str, eos: bool = False) -> List[int]:
        toks = [self._model.bos_id(), *self._model.encode(s)]
        if eos:
            toks.append(self.eos_id)
        return toks

    @property
    def eos_id(self) -> int:
        return self._model.eos_id()

    def decode(self, t: List[int]) -> str:
        out = self._model.decode(t)
        if t and self._model.id_to_piece(t[0])[0] == self._sep:
            return " " + out
        return out

    @property
    def vocab_size(self) -> int:
        return self._model.vocab_size()

def load_model(folder: str, dtype=mx.float16):
    model_path = Path(folder)
    tokenizer = Tokenizer(str(model_path / "tokenizer.model"))
    with open(model_path / "params.json", "r") as f:
        config = json.loads(f.read())
        if config.get("vocab_size", -1) < 0:
            config["vocab_size"] = tokenizer.vocab_size
        model_args = ModelArgs(**config)
    weights = mx.load(str(model_path / "weights.npz"))
    weights = tree_unflatten(list(weights.items()))
    weights = tree_map(lambda p: p.astype(dtype), weights)
    model = Model(model_args)
    model.update(weights)
    return model, tokenizer

print('Loading model...')
model, tokenizer = load_model('mlx-mistral-7B-v0.1')
model.freeze()
for l in model.layers[-8:]:
    l.attention.wq = LoRALinear.from_linear(l.attention.wq)
    l.attention.wv = LoRALinear.from_linear(l.attention.wv)
p = sum(v.size for _, v in tree_flatten(model.parameters())) / 10**6
print(f"Total parameters {p:.3f}M")
p = sum(v.size for _, v in tree_flatten(model.trainable_parameters())) / 10**6
print(f"Trainable parameters {p:.3f}M")

model.load_weights('mlx-mistral-7B-v0.1/adapters.8.npz')

# def generate(model, prompt, tokenizer):
#     prompt = mx.array(tokenizer.encode(prompt))
#
#     def generate_step():
#         temp = 0.8
#
#         def sample(logits):
#             if temp == 0:
#                 return mx.argmax(logits, axis=-1)
#             else:
#                 return mx.random.categorical(logits * (1 / temp))
#
#         logits, cache = model(prompt[None])
#         y = sample(logits[:, -1, :])
#         yield y
#
#         while True:
#             logits, cache = model(y[:, None], cache)
#             y = sample(logits.squeeze(1))
#             yield y
#     tokens = []
#     for token, _ in zip(generate_step(), range(600)):
#         tokens.append(token)
#         if token == tokenizer.eos_id:
#             break
#     mx.eval(tokens)
#     s = tokenizer.decode([t.item() for t in tokens])
#     return s

def generate(model, prompt, tokenizer, max_length=600):
    prompt = mx.array(tokenizer.encode(prompt))
    temp = 0.8

    def sample(logits):
        return mx.argmax(logits, axis=-1) if temp == 0 else mx.random.categorical(logits * (1 / temp))

    tokens = []
    logits, cache = model(prompt[None])
    y = sample(logits[:, -1, :])

    for _ in range(max_length):
        tokens.append(y)
        # if the last three tokens are 1867, 28713, 28767 then break
        if y == tokenizer.eos_id or (len(tokens) > 2 and tokenizer.decode([t.item() for t in tokens[-5:]]).__contains__('</s>')):
            break

        logits, cache = model(y[:, None], cache)
        y = sample(logits.squeeze(1))

    s = tokenizer.decode([t.item() for t in tokens])
    return s

def chat_completion(message):
    return generate(model, message, tokenizer)

# c = chat_completion("The quick brown fox jumps over the lazy dog")
