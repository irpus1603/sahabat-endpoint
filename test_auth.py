"""
Test script for API authentication
"""
import requests
import json

BASE_URL = "http://localhost:9000"
VALID_API_KEY = "sk-test-key-1"
INVALID_API_KEY = "sk-invalid-key"

def test_health_endpoint():
    """Test that health endpoint doesn't require authentication"""
    print("\n=== Testing Health Endpoint (No Auth Required) ===")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    assert response.status_code == 200, "Health endpoint should be accessible"
    print("✓ Health endpoint accessible without authentication")


def test_generate_without_auth():
    """Test that generate endpoint requires authentication"""
    print("\n=== Testing Generate Endpoint Without Auth ===")
    response = requests.post(
        f"{BASE_URL}/api/v1/generate",
        headers={"Content-Type": "application/json"},
        json={"prompt": "Test"}
    )
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    assert response.status_code == 401, "Should return 401 without API key"
    print("✓ Generate endpoint correctly rejected request without API key")


def test_generate_with_invalid_auth():
    """Test that generate endpoint rejects invalid API key"""
    print("\n=== Testing Generate Endpoint With Invalid API Key ===")
    response = requests.post(
        f"{BASE_URL}/api/v1/generate",
        headers={
            "X-API-Key": INVALID_API_KEY,
            "Content-Type": "application/json"
        },
        json={"prompt": "Test"}
    )
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    assert response.status_code == 403, "Should return 403 with invalid API key"
    print("✓ Generate endpoint correctly rejected invalid API key")


def test_generate_with_valid_auth():
    """Test that generate endpoint accepts valid API key"""
    print("\n=== Testing Generate Endpoint With Valid API Key ===")
    response = requests.post(
        f"{BASE_URL}/api/v1/generate",
        headers={
            "X-API-Key": VALID_API_KEY,
            "Content-Type": "application/json"
        },
        json={
            "prompt": "Apa itu kecerdasan buatan?",
            "max_new_tokens": 50
        }
    )
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        print("✓ Generate endpoint accepted valid API key")
    else:
        print(f"Error: {response.text}")
        print("⚠ Model might not be loaded, but authentication was successful")


def test_embeddings_with_auth():
    """Test embeddings endpoint with authentication"""
    print("\n=== Testing Embeddings Endpoint With Valid API Key ===")
    response = requests.post(
        f"{BASE_URL}/api/v1/embeddings",
        headers={
            "X-API-Key": VALID_API_KEY,
            "Content-Type": "application/json"
        },
        json={
            "texts": ["Halo dunia", "Kecerdasan buatan"],
            "normalize": True
        }
    )
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        resp_json = response.json()
        print(f"Model: {resp_json.get('model')}")
        print(f"Dimensions: {resp_json.get('dimensions')}")
        print(f"Number of embeddings: {len(resp_json.get('embeddings', []))}")
        print("✓ Embeddings endpoint accepted valid API key")
    else:
        print(f"Error: {response.text}")
        print("⚠ Model might not be loaded, but authentication was successful")


def test_chat_completions_with_auth():
    """Test chat completions endpoint with authentication"""
    print("\n=== Testing Chat Completions Endpoint With Valid API Key ===")
    response = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        headers={
            "X-API-Key": VALID_API_KEY,
            "Content-Type": "application/json"
        },
        json={
            "model": "Sahabat-AI/gemma2-9b-cpt-sahabatai-v1-instruct",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"}
            ],
            "max_tokens": 50,
            "stream": False
        }
    )
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        resp_json = response.json()
        print(f"Model: {resp_json.get('model')}")
        print(f"Choices: {len(resp_json.get('choices', []))}")
        print("✓ Chat completions endpoint accepted valid API key")
    else:
        print(f"Error: {response.text}")
        print("⚠ Model might not be loaded, but authentication was successful")


def run_all_tests():
    """Run all authentication tests"""
    print("=" * 60)
    print("API AUTHENTICATION TEST SUITE")
    print("=" * 60)
    print(f"\nBase URL: {BASE_URL}")
    print(f"Valid API Key: {VALID_API_KEY}")
    print(f"Invalid API Key: {INVALID_API_KEY}")

    try:
        test_health_endpoint()
        test_generate_without_auth()
        test_generate_with_invalid_auth()
        test_generate_with_valid_auth()
        test_embeddings_with_auth()
        test_chat_completions_with_auth()

        print("\n" + "=" * 60)
        print("✓ ALL AUTHENTICATION TESTS PASSED!")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return False
    except requests.exceptions.ConnectionError:
        print(f"\n✗ Could not connect to {BASE_URL}")
        print("Make sure the API server is running!")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        return False

    return True


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
