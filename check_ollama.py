"""Simple script to check Ollama connectivity."""

import sys
import time
from threading import Thread

print("=" * 80)
print("OLLAMA CONNECTIVITY CHECK")
print("=" * 80)

# Test 1: Check if we can import ollama
print("\n[Test 1] Checking if ollama package is installed...")
try:
    import ollama
    print("[OK] ollama package is installed")
except ImportError:
    print("[FAIL] ollama package not installed")
    print("Install with: pip install ollama")
    sys.exit(1)

# Test 2: Check if Ollama server is accessible
print("\n[Test 2] Checking if Ollama server is accessible...")
try:
    # Try to list models (simple API call)
    models = ollama.list()
    print(f"[OK] Ollama server is accessible at http://localhost:11434")
    if models.get("models"):
        print(f"[OK] Found {len(models['models'])} model(s):")
        for model in models["models"]:
            # Try different possible keys for model name
            name = model.get("name") or model.get("model") or model.get("id") or "unknown"
            size = model.get("size", 0)
            if size:
                size_gb = size / (1024**3)  # Convert to GB
                print(f"     - {name} ({size_gb:.2f} GB)")
            else:
                print(f"     - {name}")
            # Print full model dict for debugging
            print(f"       Full info: {model}")
        
        # Check if our model is available
        model_names = []
        for m in models["models"]:
            name = m.get("name") or m.get("model") or m.get("id") or ""
            if name:
                model_names.append(name)
        
        if "qwen2.5-coder:3b" in model_names:
            print(f"[OK] Required model 'qwen2.5-coder:3b' is available")
        else:
            print(f"[WARNING] Required model 'qwen2.5-coder:3b' NOT found")
            print("         Available models: " + ", ".join(model_names) if model_names else "none")
            print("         Pull it with: 'ollama pull qwen2.5-coder:3b'")
            # Don't exit - Ollama will auto-pull if needed
    else:
        print("[WARNING] No models found")
except Exception as e:
    print(f"[FAIL] Cannot connect to Ollama server: {e}")
    print("\nTroubleshooting:")
    print("1. Make sure Ollama is running: 'ollama serve'")
    print("2. Check if port 11434 is accessible")
    print("3. Try restarting Ollama")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Try a simple LLM call
print("\n[Test 3] Testing LLM response (this may take 10-30 seconds)...")
print("         Sending simple test request to qwen2.5-coder:3b")

result = {"response": None, "error": None, "completed": False, "time": 0}

def test_call():
    try:
        start = time.time()
        response = ollama.chat(
            model="qwen2.5-coder:3b",
            messages=[{"role": "user", "content": "Say 'OK' if you can read this."}],
            options={"temperature": 0.1, "num_predict": 10}
        )
        elapsed = time.time() - start
        result["response"] = response["message"]["content"]
        result["time"] = elapsed
        result["completed"] = True
    except Exception as e:
        result["error"] = str(e)
        result["completed"] = True

thread = Thread(target=test_call)
thread.daemon = True
thread.start()

# Show progress while waiting
print("         Waiting for response...", end="", flush=True)
for i in range(60):
    if result["completed"]:
        break
    time.sleep(1)
    if i % 5 == 0:
        print(f" {i}s", end="", flush=True)

thread.join(timeout=120)  # 120 second timeout (2 minutes for first load)

if not result["completed"]:
    print(f"\n\n[FAIL] LLM call timed out after 120 seconds")
    print("\nPossible causes:")
    print("1. Model is loading for the first time (can take 2-3 minutes on CPU)")
    print("2. System is low on memory (check Task Manager)")
    print("3. Ollama server is stuck (try restarting)")
    print("\nTry manually in another terminal:")
    print("  'ollama run qwen2.5-coder:3b \"Hello\"'")
    print("\nIf that works, the model is fine - just slow. You can:")
    print("1. Wait longer (increase timeout)")
    print("2. Continue with the test anyway (it will be slow)")
    print("\n[INFO] Model is available, but first load can be very slow on CPU")
    response = input("\nContinue anyway? (y/n): ").strip().lower()
    if response != 'y':
        sys.exit(1)
elif result["error"]:
    print(f"\n[FAIL] LLM call failed: {result['error']}")
    print("\nTroubleshooting:")
    print("1. Make sure model is pulled: 'ollama pull qwen2.5-coder:3b'")
    print("2. Test manually: 'ollama run qwen2.5-coder:3b \"Hello\"'")
    sys.exit(1)
else:
    print(f"[OK] LLM is responding! (took {result['time']:.1f} seconds)")
    print(f"     Response: {result['response'][:100]}...")

print("\n" + "=" * 80)
print("ALL CHECKS PASSED - Ollama is ready!")
print("=" * 80)
