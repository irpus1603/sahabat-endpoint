"""
Model loading and management
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from typing import Optional, Dict, Any
from config import get_settings
from logger import setup_logger

settings = get_settings()
logger = setup_logger(__name__)


class ModelManager:
    """Manages model loading and inference"""

    def __init__(self):
        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.embedding_model: Optional[SentenceTransformer] = None
        self.device = self._get_device()
        logger.info(f"Using device: {self.device}")

    def _get_device(self) -> str:
        """Determine the best available device"""
        if settings.DEVICE == "cuda" and torch.cuda.is_available():
            return "cuda"
        elif settings.DEVICE == "mps" and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def load_model(self) -> None:
        """Load the Sahabat-9B model and tokenizer"""
        try:
            logger.info(f"Loading model: {settings.MODEL_NAME}")

            # Configure quantization if needed
            quantization_config = None
            if settings.LOAD_IN_4BIT:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            elif settings.LOAD_IN_8BIT:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                settings.MODEL_NAME,
                trust_remote_code=True
            )

            # Load model
            model_kwargs: Dict[str, Any] = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if self.device != "cpu" else torch.float32,
            }

            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
                model_kwargs["device_map"] = "auto"
            else:
                model_kwargs["device_map"] = self.device

            self.model = AutoModelForCausalLM.from_pretrained(
                settings.MODEL_NAME,
                **model_kwargs
            )

            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def load_embedding_model(self) -> None:
        """Load the embedding model for RAG"""
        try:
            logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
            self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
            if self.device != "cpu":
                self.embedding_model = self.embedding_model.to(self.device)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        repetition_penalty: float = 1.1,
    ) -> Dict[str, Any]:
        """Generate text from prompt"""
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded")

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            input_length = inputs.input_ids.shape[1]

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=do_sample,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            tokens_generated = outputs[0].shape[0] - input_length

            # Extract only the new text (remove the prompt)
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()

            return {
                "generated_text": generated_text,
                "tokens_generated": tokens_generated,
                "finish_reason": "length" if tokens_generated >= max_new_tokens else "stop"
            }

        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            raise

    def get_embeddings(self, texts: list[str], normalize: bool = True) -> list[list[float]]:
        """Generate embeddings for texts"""
        if not self.embedding_model:
            raise RuntimeError("Embedding model not loaded")

        try:
            embeddings = self.embedding_model.encode(
                texts,
                normalize_embeddings=normalize,
                convert_to_numpy=True
            )
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            raise

    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None and self.tokenizer is not None


# Global model manager instance
model_manager = ModelManager()
