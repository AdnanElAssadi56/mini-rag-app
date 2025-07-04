from ..LLMInterface import LLMInterface
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import torch
import logging

class HuggingFaceProvider(LLMInterface):
    def __init__(
        self,
        device: str = None,
        default_input_max_characters: int = 1000,
        default_generation_max_output_tokens: int = 100,
        default_generation_temperature: float = 0.7
    ):
        # Determine device: GPU if available, else CPU
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(__name__)

        # Defaults
        self.default_input_max_characters = default_input_max_characters
        self.default_generation_max_output_tokens = default_generation_max_output_tokens
        self.default_generation_temperature = default_generation_temperature

        # Model placeholders
        self.generation_tokenizer = None
        self.generation_model = None
        self.embedding_model = None
        self.embedding_size = None

    def set_generation_model(self, model_id: str):
        """
        Load a causal language model and its tokenizer from Hugging Face Hub.
        """
        try:
            self.generation_tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.generation_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=torch.float16 if self.device.startswith("cuda") else torch.float32
            )
            # Move model to device
            self.generation_model.to(self.device)
        except Exception as e:
            self.logger.error(f"Failed to load generation model '{model_id}': {e}")
            raise

    def set_embedding_model(self, model_id: str, embedding_size: int = None):
        """
        Load a sentence-transformers model for embeddings.
        embedding_size is for interface compatibility.
        """
        try:
            self.embedding_model = SentenceTransformer(model_id, device=self.device)
            self.embedding_size = embedding_size
        except Exception as e:
            self.logger.error(f"Failed to load embedding model '{model_id}': {e}")
            raise

    def process_text(self, text: str) -> str:
        """Basic preprocessing before sending prompts or embeddings."""
        return text.strip()[: self.default_input_max_characters]

    def generate_text(
        self,
        prompt: str,
        chat_history: list = None,
        max_output_tokens: int = None,
        temperature: float = None
    ) -> str:
        """
        Generate text using the loaded causal language model.
        """
        if not self.generation_model or not self.generation_tokenizer:
            self.logger.error("Generation model or tokenizer not set")
            return None

        # Prepare parameters
        max_tokens = max_output_tokens or self.default_generation_max_output_tokens
        temp = temperature or self.default_generation_temperature

        # Construct input text
        text = self.construct_prompt(prompt, role="user")['content']
        inputs = self.generation_tokenizer(
            text,
            return_tensors="pt"
        ).to(self.device)

        outputs = self.generation_model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temp,
            pad_token_id=self.generation_tokenizer.eos_token_id
        )

        # Decode and return newly generated part
        decoded = self.generation_tokenizer.decode(
            outputs[0], skip_special_tokens=True
        )
        # Optionally trim to only generated beyond prompt
        return decoded

    def embed_text(self, text: str, document_type: str = None) -> list:
        """
        Compute embedding for the given text using the sentence-transformers model.
        """
        if not self.embedding_model:
            self.logger.error("Embedding model not set")
            return None

        processed = self.process_text(text)
        embeddings = self.embedding_model.encode(
            processed,
            convert_to_numpy=True
        )
        return embeddings.tolist()

    def construct_prompt(self, prompt: str, role: str) -> dict:
        """Wrap prompt and role into the standard message dict."""
        return {
            "role": role,
            "content": self.process_text(prompt)
        }
