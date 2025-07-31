from .Gemini import Gemini
from .Vicuna import Vicuna
from .GPT import GPT
from .Llama import Llama
from .DeepSeek import DeepSeek
import json

def load_json(file_path):
    with open(file_path) as file:
        results = json.load(file)
    return results

def create_model(config_path):
    """
    Factory method to create a LLM instance
    """
    config = load_json(config_path)

    provider = config["model_info"]["provider"].lower()
    if provider == 'gemini':
        model = Gemini(config)
    elif provider == 'vicuna':
        model = Vicuna(config)
    elif provider == 'gpt':
        model = GPT(config)
    elif provider == 'llama':
        model = Llama(config)
    elif provider == 'deepseek':
        model = DeepSeek(config)
    else:
        raise ValueError(f"ERROR: Unknown provider {provider}")
    return model