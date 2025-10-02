# Sahabat-9B API Endpoint

A robust and maintainable FastAPI application for serving the Sahabat-9B language model with RAG (Retrieval-Augmented Generation) capabilities.

## Features

- 🚀 **FastAPI** framework for high performance
- 🤖 **Sahabat-9B** model integration
- 📚 **RAG capabilities** with document retrieval and embeddings
- 🏥 **Health checks** for monitoring
- 📊 **Request logging** and error handling
- ⚙️ **Configurable** via environment variables
- 🔧 **Easy to maintain** with modular architecture

## Project Structure

```
sahabat-endpoint/
├── main.py              # FastAPI application and endpoints
├── config.py            # Configuration management
├── models.py            # Pydantic models for validation
├── model_manager.py     # Model loading and inference
├── rag_utils.py         # RAG utilities
├── logger.py            # Logging configuration
├── requirements.txt     # Python dependencies
├── .env.example         # Example environment variables
└── README.md            # This file
```

## Installation

1. **Clone the repository** (or navigate to the project directory):

```bash
cd /Users/supriyadi/Projects/LLM/sahabat-endpoint
```

2. **Create a virtual environment**:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:

```bash
pip install -r requirements.txt
```

4. **Configure environment variables**:

```bash
cp .env.example .env
# Edit .env with your settings
```

## Configuration

Key configuration options in `.env`:

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_NAME` | HuggingFace model name | `LocalDoc/Sahabat-9B` |
| `DEVICE` | Device to use | `cuda` |
| `LOAD_IN_4BIT` | Use 4-bit quantization | `false` |
| `LOAD_IN_8BIT` | Use 8-bit quantization | `false` |
| `PORT` | API server port | `8000` |
| `LOG_LEVEL` | Logging level | `INFO` |

## Usage

### Start the server

```bash
python main.py
```

Or with uvicorn directly:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### API Documentation

Once running, access the interactive API docs at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API Endpoints

### Health Check

```bash
GET /health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0",
  "device": "cuda"
}
```

### Text Generation

```bash
POST /api/v1/generate
```

Request:
```json
{
  "prompt": "Apa itu kecerdasan buatan?",
  "max_new_tokens": 256,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 50
}
```

Response:
```json
{
  "generated_text": "Kecerdasan buatan adalah...",
  "prompt": "Apa itu kecerdasan buatan?",
  "tokens_generated": 145,
  "finish_reason": "stop"
}
```

### Generate Embeddings

```bash
POST /api/v1/embeddings
```

Request:
```json
{
  "texts": ["Halo dunia", "Kecerdasan buatan"],
  "normalize": true
}
```

Response:
```json
{
  "embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...]],
  "model": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
  "dimensions": 768
}
```

### Chunk Text

```bash
POST /api/v1/chunk
```

Request:
```json
{
  "text": "Long document text here...",
  "chunk_size": 512,
  "chunk_overlap": 50
}
```

Response:
```json
{
  "chunks": ["chunk 1...", "chunk 2..."],
  "num_chunks": 2
}
```

### RAG Query

```bash
POST /api/v1/rag
```

Request:
```json
{
  "query": "Apa itu pembelajaran mesin?",
  "documents": [
    "Pembelajaran mesin adalah cabang dari kecerdasan buatan.",
    "Python adalah bahasa pemrograman yang populer."
  ],
  "top_k": 2,
  "max_new_tokens": 512
}
```

Response:
```json
{
  "answer": "Berdasarkan dokumen...",
  "retrieved_documents": ["Pembelajaran mesin adalah..."],
  "relevance_scores": [0.95, 0.72]
}
```

## Examples

### cURL Examples

```bash
# Health check
curl http://localhost:8000/health

# Generate text
curl -X POST http://localhost:8000/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Jelaskan tentang Python:",
    "max_new_tokens": 200,
    "temperature": 0.7
  }'

# RAG query
curl -X POST http://localhost:8000/api/v1/rag \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Apa keuntungan menggunakan FastAPI?",
    "documents": [
      "FastAPI adalah framework web modern untuk Python.",
      "FastAPI sangat cepat dan mudah digunakan.",
      "Django adalah framework web Python yang populer."
    ],
    "top_k": 2
  }'
```

### Python Client Example

```python
import requests

# Generate text
response = requests.post(
    "http://localhost:8000/api/v1/generate",
    json={
        "prompt": "Apa itu kecerdasan buatan?",
        "max_new_tokens": 256,
        "temperature": 0.7
    }
)
print(response.json())

# RAG query
response = requests.post(
    "http://localhost:8000/api/v1/rag",
    json={
        "query": "Bagaimana cara kerja neural network?",
        "documents": [
            "Neural network terinspirasi dari otak manusia.",
            "Python adalah bahasa pemrograman.",
            "Deep learning menggunakan neural network berlapis."
        ],
        "top_k": 2
    }
)
print(response.json())
```

## Performance Optimization

### GPU Memory Management

For limited VRAM, enable quantization:

```bash
# 4-bit quantization (recommended for <16GB VRAM)
LOAD_IN_4BIT=true

# 8-bit quantization (recommended for <24GB VRAM)
LOAD_IN_8BIT=true
```

### CPU Inference

For CPU-only systems:

```bash
DEVICE=cpu
```

### Mac with Apple Silicon

For M1/M2/M3 Macs:

```bash
DEVICE=mps
```

## Monitoring

The API includes:
- Health check endpoint at `/health`
- Request timing in `X-Process-Time` header
- Structured JSON logging
- Automatic error handling and reporting

## Development

### Run in debug mode

```bash
DEBUG=true python main.py
```

### Run tests (add your tests)

```bash
pytest tests/
```

## Production Deployment

### Using Docker (create Dockerfile)

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Using systemd

Create `/etc/systemd/system/sahabat-api.service`:

```ini
[Unit]
Description=Sahabat-9B API
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/path/to/sahabat-endpoint
Environment="PATH=/path/to/venv/bin"
ExecStart=/path/to/venv/bin/python main.py
Restart=always

[Install]
WantedBy=multi-user.target
```

## Troubleshooting

### Model loading issues

- Check if model name is correct
- Ensure sufficient disk space for model download
- Verify CUDA/GPU drivers if using GPU

### Out of memory errors

- Enable quantization (4-bit or 8-bit)
- Reduce `MODEL_MAX_LENGTH`
- Use CPU instead of GPU

### Slow inference

- Use GPU instead of CPU
- Enable quantization for faster loading
- Reduce `max_new_tokens` in requests

## License

MIT License

## Support

For issues and questions:
- Check the API docs at `/docs`
- Review logs for error details
- Open an issue on the repository
