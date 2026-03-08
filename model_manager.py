"""
Model loading and management
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
from sentence_transformers import SentenceTransformer
from typing import Optional, Dict, Any, Generator
from threading import Thread
from config import get_settings
from logger import setup_logger
from typing import Optional, Dict, Any, Generator

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    Llama = None  # type: ignore[assignment,misc]
    LLAMA_CPP_AVAILABLE = False

settings = get_settings()
logger = setup_logger(__name__)


class ModelManager:
    """Manages model loading and inference"""

    def __init__(self):
        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.llm = None  # llama-cpp Llama instance for GGUF backend
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
        """Load the LLM model — supports 'transformers' and 'gguf' backends"""
        if settings.MODEL_BACKEND == "gguf":
            self._load_gguf_model()
        else:
            self._load_transformers_model()

    def _load_gguf_model(self) -> None:
        """Load a GGUF model via llama-cpp-python"""
        if not LLAMA_CPP_AVAILABLE or Llama is None:
            raise RuntimeError(
                "llama-cpp-python is not installed. Install it with: pip install llama-cpp-python"
            )
        try:
            logger.info(f"Loading GGUF model: {settings.MODEL_NAME} (file: {settings.MODEL_FILE})")
            n_gpu_layers = settings.N_GPU_LAYERS
            self.llm = Llama.from_pretrained(
                repo_id=settings.MODEL_NAME,
                filename=settings.MODEL_FILE,
                n_ctx=settings.N_CTX,
                n_gpu_layers=n_gpu_layers,
                verbose=False,
            )
            logger.info("GGUF model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load GGUF model: {str(e)}")
            raise

    def _load_transformers_model(self) -> None:
        """Load a HuggingFace transformers model (original behaviour)"""
        try:
            logger.info(f"Loading model: {settings.MODEL_NAME}")

            # Prepare authentication token
            token = settings.HUGGINGFACE_TOKEN if settings.HUGGINGFACE_TOKEN else None
            if token:
                logger.info("Using HuggingFace authentication token")

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
                trust_remote_code=True,
                token=token,
                use_fast=True  # Use fast tokenizer for better performance
            )

            # Set padding token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Override tokenizer's default model_max_length (often 2048)
            # so it matches the actual context window configured in .env
            if self.tokenizer:
                self.tokenizer.model_max_length = settings.MODEL_MAX_LENGTH

            # Load model
            model_kwargs: Dict[str, Any] = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if self.device != "cpu" else torch.float32,
                "token": token,
                "low_cpu_mem_usage": True,  # Reduce CPU memory during loading
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

            # Enable eval mode for inference (disables dropout)
            self.model.eval()

            # Enable Flash Attention 2 if available (significantly faster)
            if hasattr(self.model.config, 'attn_implementation'):
                try:
                    self.model.config.attn_implementation = "flash_attention_2"
                    logger.info("Flash Attention 2 enabled")
                except Exception:
                    logger.warning("Flash Attention 2 not available, using default attention")

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

    # ------------------------------------------------------------------
    # GGUF backend helpers
    # ------------------------------------------------------------------

    def _generate_gguf(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> Dict[str, Any]:
        if self.llm is None:
            raise RuntimeError("GGUF model not loaded")
        try:
            output = self.llm(
                prompt,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                echo=False,
            )
            generated_text = output["choices"][0]["text"].strip()
            tokens_generated = output["usage"]["completion_tokens"]
            finish_reason = output["choices"][0]["finish_reason"] or "stop"
            return {
                "generated_text": generated_text,
                "tokens_generated": tokens_generated,
                "finish_reason": finish_reason,
            }
        except Exception as e:
            logger.error(f"GGUF generation failed: {str(e)}")
            raise

    def _chat_completion_gguf(
        self,
        messages: list,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
    ) -> Dict[str, Any]:
        if self.llm is None:
            raise RuntimeError("GGUF model not loaded")
        try:
            output = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repetition_penalty,
                stream=False,
            )
            choice = output["choices"][0]
            usage = output["usage"]
            return {
                "generated_text": choice["message"]["content"],
                "prompt_tokens": usage["prompt_tokens"],
                "completion_tokens": usage["completion_tokens"],
                "total_tokens": usage["total_tokens"],
                "finish_reason": choice["finish_reason"] or "stop",
            }
        except Exception as e:
            logger.error(f"GGUF chat completion failed: {str(e)}")
            raise

    def _chat_completion_stream_gguf(
        self,
        messages: list,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
    ) -> Generator[str, None, None]:
        if self.llm is None:
            raise RuntimeError("GGUF model not loaded")
        try:
            stream = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repetition_penalty,
                stream=True,
            )
            for chunk in stream:
                delta = chunk["choices"][0]["delta"]
                text = delta.get("content", "")
                if text:
                    yield text
        except Exception as e:
            logger.error(f"GGUF streaming failed: {str(e)}")
            raise

    # ------------------------------------------------------------------
    # Public inference methods
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = settings.DEFAULT_MAX_NEW_TOKENS,
        temperature: float = settings.DEFAULT_TEMPERATURE,
        top_p: float = settings.DEFAULT_TOP_P,
        top_k: int = settings.DEFAULT_TOP_K,
        do_sample: bool = True,
        repetition_penalty: float = 1.1,
    ) -> Dict[str, Any]:
        """Generate text from prompt"""
        if settings.MODEL_BACKEND == "gguf":
            return self._generate_gguf(prompt, max_new_tokens, temperature, top_p, top_k)
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
                    use_cache=True,  # Enable KV cache for faster generation
                    num_beams=1,  # Greedy decoding is faster
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

    def chat_completion_stream(
        self,
        messages: list[dict],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = 1.1,
    ) -> Generator[str, None, None]:
        """
        Generate streaming chat completion from messages

        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling
            top_k: Top-k sampling
            repetition_penalty: Repetition penalty

        Yields:
            Generated text tokens as they are produced
        """
        if settings.MODEL_BACKEND == "gguf":
            yield from self._chat_completion_stream_gguf(
                messages=messages,
                max_tokens=max_tokens if max_tokens is not None else settings.DEFAULT_MAX_NEW_TOKENS,
                temperature=temperature if temperature is not None else settings.DEFAULT_TEMPERATURE,
                top_p=top_p if top_p is not None else settings.DEFAULT_TOP_P,
                top_k=top_k if top_k is not None else settings.DEFAULT_TOP_K,
                repetition_penalty=repetition_penalty or 1.1,
            )
            return
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded")

        try:
            # Format messages into a prompt
            if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                # Manual formatting for Gemma2
                prompt = ""
                for msg in messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")

                    if role == "system":
                        prompt += f"<start_of_turn>user\n{content}<end_of_turn>\n"
                    elif role == "user":
                        prompt += f"<start_of_turn>user\n{content}<end_of_turn>\n"
                    elif role == "assistant":
                        prompt += f"<start_of_turn>model\n{content}<end_of_turn>\n"

                prompt += "<start_of_turn>model\n"

            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            # Create streamer
            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )

            # Generation kwargs
            generation_kwargs = {
                **inputs,
                "max_new_tokens": max_tokens if max_tokens is not None else settings.DEFAULT_MAX_NEW_TOKENS,
                "temperature": temperature if temperature is not None else settings.DEFAULT_TEMPERATURE,
                "top_p": top_p if top_p is not None else settings.DEFAULT_TOP_P,
                "top_k": top_k if top_k is not None else settings.DEFAULT_TOP_K,
                "do_sample": True,
                "repetition_penalty": repetition_penalty,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "use_cache": True,
                "num_beams": 1,
                "streamer": streamer,
            }

            # Start generation in a separate thread
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()

            # Yield tokens as they are generated
            for text in streamer:
                yield text

            thread.join()

        except Exception as e:
            logger.error(f"Streaming chat completion failed: {str(e)}")
            raise

    def chat_completion(
        self,
        messages: list[dict],
        max_tokens: int = settings.DEFAULT_MAX_NEW_TOKENS,
        temperature: float = settings.DEFAULT_TEMPERATURE,
        top_p: float = settings.DEFAULT_TOP_P,
        top_k: int = settings.DEFAULT_TOP_K,
        repetition_penalty: float = 1.1,
        stream: bool = False,
        stream_options: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Generate chat completion from messages

        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling
            top_k: Top-k sampling
            repetition_penalty: Repetition penalty
            stream: Whether to stream the response
            stream_options: Stream options

        Returns:
            Dict with generated_text, prompt_tokens, completion_tokens, finish_reason
        """
        if settings.MODEL_BACKEND == "gguf":
            return self._chat_completion_gguf(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
            )
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded")

        try:
            # Format messages into a prompt using Gemma2 chat template
            # If tokenizer has chat template, use it; otherwise format manually
            if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                # Manual formatting for Gemma2
                prompt = ""
                for msg in messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")

                    if role == "system":
                        prompt += f"<start_of_turn>user\n{content}<end_of_turn>\n"
                    elif role == "user":
                        prompt += f"<start_of_turn>user\n{content}<end_of_turn>\n"
                    elif role == "assistant":
                        prompt += f"<start_of_turn>model\n{content}<end_of_turn>\n"

                # Add generation prompt
                prompt += "<start_of_turn>model\n"

            # Tokenize and generate
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            input_length = inputs.input_ids.shape[1]

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=True,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,  # Enable KV cache for faster generation
                    num_beams=1,  # Greedy decoding is faster than beam search
                )

            # Decode the generated text
            full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            output_length = outputs[0].shape[0]
            tokens_generated = output_length - input_length

            # Extract only the assistant's response
            # Remove the prompt part
            if full_text.startswith(prompt):
                generated_text = full_text[len(prompt):].strip()
            else:
                # Fallback: try to extract after the last model turn marker
                if "<start_of_turn>model\n" in full_text:
                    parts = full_text.split("<start_of_turn>model\n")
                    generated_text = parts[-1].strip()
                    # Remove end marker if present
                    if "<end_of_turn>" in generated_text:
                        generated_text = generated_text.split("<end_of_turn>")[0].strip()
                else:
                    generated_text = full_text.strip()

            return {
                "generated_text": generated_text,
                "prompt_tokens": input_length,
                "completion_tokens": tokens_generated,
                "total_tokens": output_length,
                "finish_reason": "length" if tokens_generated >= max_tokens else "stop"
            }

        except Exception as e:
            logger.error(f"Chat completion failed: {str(e)}")
            raise

    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        if settings.MODEL_BACKEND == "gguf":
            return self.llm is not None
        return self.model is not None and self.tokenizer is not None


# Global model manager instance
model_manager = ModelManager()
