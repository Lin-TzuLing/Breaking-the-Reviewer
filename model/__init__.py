from .gpt_4o import GPT4o
from .gpt_4o_mini import GPT4oMini
from .llama_70b import Llama70B
from .mistral_small_3 import MistralSmall3

MODELS = {
    "gpt-4o-mini": GPT4oMini,
    "gpt-4o": GPT4o,
    "llama-3.3-70b": Llama70B,
    "mistral-small-3.1": MistralSmall3,
}


def load_model(config):
    config = vars(config)
    model = MODELS[config["model_name"].lower()](**config)
    return model
