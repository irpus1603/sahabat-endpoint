# Performance Optimization Guide

This guide explains how to optimize the Sahabat-9B API for faster inference.

## Quick Performance Checklist

### ✅ Already Implemented

The following optimizations are already enabled in the codebase:

1. **KV Cache** - Caches key-value pairs for faster autoregressive generation
2. **Fast Tokenizer** - Uses Rust-based tokenizers for faster processing
3. **Torch No Grad** - Disables gradient computation during inference
4. **Eval Mode** - Disables dropout and batch normalization training behavior
5. **Low CPU Memory** - Optimizes memory usage during model loading

### 🚀 Recommended Hardware Optimizations

#### 1. **Use Quantization (Best for Limited VRAM)**

**4-bit Quantization (Recommended for <16GB VRAM):**
```bash
# In .env file
LOAD_IN_4BIT=true
```
- Reduces memory by ~75%
- Minimal accuracy loss
- 2-3x faster inference

**8-bit Quantization (For 16-24GB VRAM):**
```bash
LOAD_IN_8BIT=true
```
- Reduces memory by ~50%
- Better accuracy than 4-bit
- 1.5-2x faster inference

#### 2. **Enable Flash Attention 2 (Requires Installation)**

Flash Attention 2 provides 2-4x speedup for long sequences.

**Installation:**
```bash
pip install flash-attn --no-build-isolation
```

**Enable in config:**
```bash
USE_FLASH_ATTENTION=true
```

**Note:** Requires:
- CUDA-capable GPU (compute capability 8.0+)
- CUDA 11.6+
- PyTorch compiled with CUDA support

#### 3. **Use Better GPU**

Performance comparison for Sahabat-9B:
- **A100 (40GB)**: ~15-20 tokens/sec (best)
- **A10 (24GB)**: ~10-15 tokens/sec
- **RTX 4090 (24GB)**: ~12-18 tokens/sec
- **RTX 3090 (24GB)**: ~8-12 tokens/sec
- **T4 (16GB)**: ~5-8 tokens/sec (with quantization)

### ⚡ Configuration Optimizations

#### 1. **Reduce max_tokens**

Lower token limits = faster responses:
```python
# Instead of
"max_tokens": 2048  # Slow

# Use
"max_tokens": 512   # 4x faster
```

#### 2. **Lower Temperature**

Lower temperature converges faster:
```python
# Instead of
"temperature": 0.9  # More creative, slower

# Use
"temperature": 0.3  # Faster, more deterministic
```

#### 3. **Disable Sampling for Speed**

For deterministic output (e.g., classification):
```python
{
  "temperature": 0.0,  # Greedy decoding
  "top_p": 1.0,
  "top_k": 1
}
```

### 🔧 System-Level Optimizations

#### 1. **Use torch.compile (PyTorch 2.0+)**

Add to `model_manager.py` after model loading:
```python
# Requires PyTorch 2.0+
if hasattr(torch, 'compile'):
    self.model = torch.compile(self.model, mode="reduce-overhead")
```

**Expected speedup:** 20-30% faster

#### 2. **Enable TensorFloat-32 (TF32)**

Add to model initialization:
```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

**Expected speedup:** 10-15% faster (on Ampere+ GPUs)

#### 3. **Optimize CUDA Settings**

```bash
# Add to environment
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### 📊 Benchmarking

Test your setup with different configurations:

```bash
# Test with default settings
time curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Sahabat-AI/gemma2-9b-cpt-sahabatai-v1-instruct",
    "messages": [{"role": "user", "content": "Test"}],
    "max_tokens": 100
  }'
```

### 🎯 Expected Performance

**With Optimizations (GPU: RTX 3090, 4-bit quantization):**
- Model loading: ~30-60 seconds
- First request (cold): ~2-3 seconds
- Subsequent requests: ~1-2 seconds
- Token generation: ~15-20 tokens/sec

**Bottlenecks to Check:**

1. **Slow first request?** → Model loading issue, use smaller quantization
2. **Slow all requests?** → Enable KV cache, use quantization, reduce max_tokens
3. **High memory usage?** → Enable 4-bit or 8-bit quantization
4. **Inconsistent speed?** → Check GPU utilization, thermal throttling

### 🐛 Debugging Slow Performance

```python
# Add timing logs in model_manager.py
import time

def chat_completion(self, ...):
    start = time.time()

    # Tokenization
    t1 = time.time()
    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
    logger.info(f"Tokenization: {time.time() - t1:.2f}s")

    # Generation
    t2 = time.time()
    outputs = self.model.generate(...)
    logger.info(f"Generation: {time.time() - t2:.2f}s")

    # Decoding
    t3 = time.time()
    text = self.tokenizer.decode(...)
    logger.info(f"Decoding: {time.time() - t3:.2f}s")

    logger.info(f"Total: {time.time() - start:.2f}s")
```

### 💡 Quick Wins

**Fastest setup for immediate results:**

1. Enable 4-bit quantization:
   ```bash
   LOAD_IN_4BIT=true
   ```

2. Reduce default max tokens:
   ```bash
   DEFAULT_MAX_NEW_TOKENS=256
   ```

3. Use lower temperature for deterministic tasks:
   ```python
   "temperature": 0.3
   ```

4. Monitor GPU usage:
   ```bash
   watch -n 1 nvidia-smi
   ```

### 📈 Advanced: Batching Multiple Requests

For high-throughput scenarios, implement batching:

```python
# In model_manager.py (advanced)
def batch_chat_completion(self, batch_messages: list):
    # Process multiple requests in parallel
    # Requires padding and attention masks
    pass
```

### 🔍 Monitoring

Add metrics to track performance:

```python
# In main.py
from prometheus_client import Counter, Histogram

inference_duration = Histogram('inference_duration_seconds', 'Inference duration')
token_throughput = Histogram('tokens_per_second', 'Token generation throughput')
```

## Summary

**Priority optimizations:**
1. ✅ **Enable 4-bit quantization** (biggest impact)
2. ✅ **Reduce max_tokens** to minimum needed
3. ✅ **Use better GPU** if available
4. ⚡ **Install Flash Attention 2**
5. ⚡ **Enable torch.compile** (PyTorch 2.0+)

**Expected Results:**
- 2-4x faster inference with quantization
- 50-70% memory reduction
- Can handle more concurrent requests
