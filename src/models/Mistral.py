import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .Model import Model

class Mistral(Model):
    def __init__(self, config):
        super().__init__(config)
        self.max_output_tokens = int(config["params"]["max_output_tokens"])
        self.device = config["params"]["device"]
        self.max_output_tokens = config["params"]["max_output_tokens"]

        # Mistral specific parameters
        self.repetition_penalty = float(config["params"].get("repetition_penalty", 1.1))
        self.do_sample = self.__str_to_bool(config["params"].get("do_sample", "true"))
        self.top_p = float(config["params"].get("top_p", 0.9))
        self.top_k = int(config["params"].get("top_k", 50))

        # Handle API key if needed
        hf_token = None
        if "api_key_info" in config and config["api_key_info"]:
            api_pos = int(config["api_key_info"]["api_key_use"])
            hf_token = config["api_key_info"]["api_keys"][api_pos]

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.name,
            use_auth_token=hf_token,
            trust_remote_code=True
        )

        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.name,
            torch_dtype=torch.float16,
            use_auth_token=hf_token,
            trust_remote_code=True,
            device_map="auto" if self.device == "auto" else None
        )

        if self.device != "auto":
            self.model = self.model.to(self.device)

    def __str_to_bool(self, s):
        if type(s) == str:
            if s.lower() == 'true':
                return True
            elif s.lower() == 'false':
                return False
        elif type(s) == bool:
            return s
        raise ValueError(f'{s} is not a valid boolean')

    def query(self, msg):
        try:
            # Tokenize input
            input_ids = self.tokenizer(
                msg,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).input_ids.to(self.model.device)

            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    temperature=self.temperature,
                    max_new_tokens=self.max_output_tokens,
                    do_sample=self.do_sample,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    repetition_penalty=self.repetition_penalty,
                    pad_token_id=self.tokenizer.eos_token_id,
                    early_stopping=True
                )

            # Decode output
            full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract only the generated part (remove input prompt)
            result = full_output[len(msg):].strip()

            return result

        except Exception as e:
            print(f"Error in Mistral query: {e}")
            return ""