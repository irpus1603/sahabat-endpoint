# API Authentication Setup

This document explains how to configure and use token-based authentication for the Sahabat-9B API.

## Overview

The API uses **X-API-Key header authentication** to secure endpoints. All endpoints except `/health` and `/` require a valid API key.

## Configuration

### 1. Update `.env` file

Add the following authentication settings to your `.env` file:

```env
# Authentication
ENABLE_API_KEY_AUTH=True
API_KEYS=["sk-test-key-1", "sk-test-key-2", "sk-your-api-key-here"]
API_KEY_HEADER=X-API-Key
```

### 2. Configuration Options

- **ENABLE_API_KEY_AUTH** (bool): Enable or disable API key authentication
  - `True`: Authentication is required (default)
  - `False`: Skip authentication (useful for development/testing)

- **API_KEYS** (list): List of valid API keys
  - Format: `["key1", "key2", "key3"]`
  - Keys can be any string (e.g., `sk-abc123def456`)

- **API_KEY_HEADER** (str): HTTP header name for API key (default: `X-API-Key`)

## Using the API

### Making Authenticated Requests

Include the API key in the request header:

```bash
curl -X POST http://localhost:9000/api/v1/generate \
  -H "X-API-Key: sk-your-api-key-here" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Apa itu kecerdasan buatan?",
    "max_new_tokens": 256
  }'
```

### Python Example

```python
import requests

API_KEY = "sk-your-api-key-here"
BASE_URL = "http://localhost:9000"

headers = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
}

# Generate text
response = requests.post(
    f"{BASE_URL}/api/v1/generate",
    headers=headers,
    json={
        "prompt": "Apa itu kecerdasan buatan?",
        "max_new_tokens": 256
    }
)

print(response.json())
```

### JavaScript/TypeScript Example

```javascript
const API_KEY = "sk-your-api-key-here";
const BASE_URL = "http://localhost:9000";

async function generateText(prompt) {
  const response = await fetch(`${BASE_URL}/api/v1/generate`, {
    method: "POST",
    headers: {
      "X-API-Key": API_KEY,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      prompt: prompt,
      max_new_tokens: 256
    })
  });

  return response.json();
}

generateText("Apa itu kecerdasan buatan?").then(console.log);
```

## Protected Endpoints

The following endpoints require API key authentication:

- `POST /api/v1/generate` - Generate text
- `POST /api/v1/embeddings` - Generate embeddings
- `POST /api/v1/chunk` - Chunk documents
- `POST /api/v1/rag` - RAG query
- `POST /v1/chat/completions` - OpenAI-compatible chat

## Public Endpoints

These endpoints do NOT require authentication:

- `GET /health` - Health check
- `GET /` - API root information

## Error Handling

### Missing API Key

**Status Code:** 401 Unauthorized

```json
{
  "detail": "Missing API key"
}
```

### Invalid API Key

**Status Code:** 403 Forbidden

```json
{
  "detail": "Invalid API key"
}
```

## Best Practices

1. **Generate Strong Keys**: Use cryptographically secure random strings for API keys
   - Example: `sk-` prefix with 32+ random characters

2. **Rotate Keys Regularly**: Update API keys periodically

3. **Environment Variables**: Never hardcode API keys in your source code
   ```python
   import os
   api_keys = os.getenv("API_KEYS", "[]")
   ```

4. **Use HTTPS in Production**: Always use HTTPS/TLS in production environments

5. **Rate Limiting**: Consider combining with rate limiting for additional protection

## Development/Testing

To disable authentication for development:

```env
ENABLE_API_KEY_AUTH=False
```

With authentication disabled, you can make requests without the API key header:

```bash
curl -X POST http://localhost:9000/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello"}'
```

## Integration with OpenAI Clients

If using OpenAI Python client or similar libraries:

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-your-api-key-here",
    base_url="http://localhost:9000"
)

response = client.chat.completions.create(
    model="Sahabat-AI/gemma2-9b-cpt-sahabatai-v1-instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
)

print(response.choices[0].message.content)
```
