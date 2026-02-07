"""
Test script for FRED Agent API
Usage: python test_api.py
Make sure the API is running (docker-compose up) before running this script.
"""

import requests
import json
import sys
import io

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

API_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint"""
    print("[TEST] Testing health endpoint...")
    response = requests.get(f"{API_URL}/health")

    if response.status_code == 200:
        data = response.json()
        print("[OK] Health check passed!")
        print(f"   Status: {data['status']}")
        print(f"   Services: {data['services']}")
        return True
    else:
        print(f"[FAIL] Health check failed: {response.status_code}")
        return False


def test_chat_stream(query: str = "¿Qué es FRED?"):
    """Test chat streaming endpoint"""
    print(f"\n[CHAT] Testing chat stream with query: '{query}'")
    print("-" * 60)

    url = f"{API_URL}/chat/stream"
    payload = {
        "query": query,
        "user_name": "TestUser"
    }

    try:
        response = requests.post(url, json=payload, stream=True, timeout=60)

        if response.status_code != 200:
            print(f"[FAIL] Request failed: {response.status_code}")
            print(response.text)
            return

        thread_id = response.headers.get("X-Thread-ID", "unknown")
        print(f"[INFO] Thread ID: {thread_id}\n")

        # Process SSE stream
        for line in response.iter_lines():
            if line:
                line_text = line.decode('utf-8')
                if line_text.startswith('data: '):
                    event_data = line_text[6:]  # Remove 'data: ' prefix
                    try:
                        event = json.loads(event_data)
                        event_type = event.get('type')
                        data = event.get('data')

                        if event_type == 'log':
                            print(f"[LOG] {data.get('message', data)}")
                        elif event_type == 'message':
                            print(f"\n[AGENT] {data.get('content', data)}\n")
                        elif event_type == 'questions':
                            print(f"\n[QUESTIONS] {len(data.get('questions', []))} questions")
                            for q in data.get('questions', []):
                                print(f"   - {q.get('question')}")
                        elif event_type == 'error':
                            print(f"[ERROR] {data}")
                        elif event_type == 'done':
                            print(f"\n[DONE] Thread: {data.get('thread_id')}")
                    except json.JSONDecodeError:
                        print(f"[WARN] Could not parse: {event_data}")

        print("-" * 60)

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Request error: {e}")
    except KeyboardInterrupt:
        print("\n[STOP] Interrupted by user")


def test_root():
    """Test root endpoint"""
    print("\n[TEST] Testing root endpoint...")
    response = requests.get(f"{API_URL}/")

    if response.status_code == 200:
        data = response.json()
        print(f"[OK] API Info:")
        print(f"   Name: {data.get('name')}")
        print(f"   Version: {data.get('version')}")
        print(f"   Docs: {data.get('docs')}")
        return True
    else:
        print(f"[FAIL] Root endpoint failed: {response.status_code}")
        return False


def main():
    print("=" * 60)
    print("FRED Agent API Test Suite")
    print("=" * 60)

    # Test 1: Root endpoint
    if not test_root():
        print("\n[WARN] API might not be running. Start it with:")
        print("   cd docker && docker-compose up")
        sys.exit(1)

    # Test 2: Health check
    if not test_health():
        print("\n[WARN] Health check failed. Check API logs.")
        sys.exit(1)

    # Test 3: Chat stream
    test_chat_stream("¿Qué es FRED y qué puede hacer?")

    print("\n" + "=" * 60)
    print("[SUCCESS] All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[STOP] Tests interrupted by user")
        sys.exit(0)
