import requests
import re
import json
from src.models import load_json


class LLMProvider:
    """Base class for LLM providers"""

    def __init__(self, config):
        self.config = config
        self.model_name = config['model_info']['name']
        self.api_keys = config['api_key_info']['api_keys']
        self.current_key_index = config['api_key_info']['api_key_use']
        self.params = config['params']

    def get_current_api_key(self):
        return self.api_keys[self.current_key_index % len(self.api_keys)]

    def rotate_api_key(self):
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)

    def query(self, prompt, return_json=False, conversation_history=None):
        raise NotImplementedError("Subclasses must implement query method")


class OpenAIProvider(LLMProvider):
    """OpenAI/ChatGPT provider"""

    def query(self, prompt, return_json=False, conversation_history=None):
        url = 'https://api.openai.com/v1/chat/completions'
        headers = {
            'Authorization': f"Bearer {self.get_current_api_key()}",
            'Content-Type': 'application/json'
        }

        # Build message history
        messages = [{'role': 'system', 'content': 'You are a helpful AI assistant.'}]

        # Add conversation history if provided
        if conversation_history:
            messages.extend(conversation_history)

        # Add current user input
        messages.append({'role': 'user', 'content': prompt})

        data = {
            'model': self.model_name,
            'temperature': self.params.get('temperature', 0.1),
            'max_tokens': self.params.get('max_output_tokens', 150),
            'messages': messages
        }

        if 'seed' in self.params:
            data['seed'] = self.params['seed']
        if return_json:
            data['response_format'] = {"type": "json_object"}

        try:
            response = requests.post(url, headers=headers, json=data, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            self.rotate_api_key()
            return None


class LocalModelProvider(LLMProvider):
    """Provider for local models (llama, mistral) via OpenAI-compatible API"""

    def __init__(self, config):
        super().__init__(config)
        self.base_url = config.get('api_base_url', 'http://localhost:8000/v1')

    def query(self, prompt, return_json=False, conversation_history=None):
        url = f'{self.base_url}/chat/completions'
        headers = {'Content-Type': 'application/json'}

        # Add API key if available
        if self.api_keys and self.api_keys[0]:
            headers['Authorization'] = f"Bearer {self.get_current_api_key()}"

        # Build message history
        messages = [{'role': 'system', 'content': 'You are a helpful AI assistant.'}]

        # Add conversation history if provided
        if conversation_history:
            messages.extend(conversation_history)

        # Add current user input
        messages.append({'role': 'user', 'content': prompt})

        data = {
            'model': self.model_name,
            'temperature': self.params.get('temperature', 0.1),
            'max_tokens': self.params.get('max_output_tokens', 150),
            'messages': messages
        }

        try:
            response = requests.post(url, headers=headers, json=data, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            return None


def load_model_config(config_file):
    """Load model configuration file"""
    with open(config_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_llm_provider(config_path):
    """
    Factory function to get the appropriate LLM provider based on config

    Args:
        config_path: Path to the configuration JSON file

    Returns:
        LLMProvider instance
    """
    config = load_json(config_path)
    provider_name = config['model_info']['provider'].lower()

    provider_map = {
        'chatgpt': OpenAIProvider,
        'openai': OpenAIProvider,
        'gpt': OpenAIProvider,
        'llama': LocalModelProvider,
        'mistral': LocalModelProvider
    }

    provider_class = provider_map.get(provider_name)
    if not provider_class:
        raise ValueError(f"Unsupported provider: {provider_name}")

    return provider_class(config)


def get_model_name_for_filename(config_path):
    """
    Extract model name for filename from config

    Args:
        config_path: Path to the configuration JSON file

    Returns:
        Cleaned model name suitable for filename
    """
    config = load_json(config_path)
    model_name = config['model_info']['name']

    # Clean up model name for filename
    model_name = model_name.replace('-', '_').replace('/', '_').replace('\\', '_')
    model_name = model_name.split(':')[0]
    model_name = re.sub(r'[^a-zA-Z0-9_]', '', model_name)

    return model_name


class LLMManager:
    """Unified LLM manager providing convenient calling interface"""

    def __init__(self, config_path):
        """
        Initialize LLM manager

        Args:
            config_path: Configuration file path
        """
        self.config_path = config_path
        self.config = load_model_config(config_path)
        self.provider = get_llm_provider(config_path)

    def query(self, prompt, return_json=False, conversation_history=None):
        """
        Unified query interface

        Args:
            prompt: Input prompt
            return_json: Whether to return JSON format
            conversation_history: Conversation history

        Returns:
            Model response content
        """
        return self.provider.query(prompt, return_json, conversation_history)

    def query_with_history(self, prompt, conversation_history=None):
        """
        Query with conversation history

        Args:
            prompt: Input prompt
            conversation_history: Conversation history list

        Returns:
            Model response content
        """
        return self.query(prompt, conversation_history=conversation_history)

    def query_json(self, prompt, conversation_history=None):
        """
        Query returning JSON format

        Args:
            prompt: Input prompt
            conversation_history: Conversation history

        Returns:
            JSON format model response
        """
        return self.query(prompt, return_json=True, conversation_history=conversation_history)

    def get_model_info(self):
        """Get model information"""
        return {
            'provider': self.config['model_info']['provider'],
            'model_name': self.config['model_info']['name'],
            'parameters': self.config['params']
        }

    def rotate_api_key(self):
        """Rotate API key"""
        self.provider.rotate_api_key()