"""Simple test to verify both versions can be imported and basic functionality works."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

print("="*80)
print("SIMPLE IMPORT TEST")
print("="*80)

# Test 1: Import LLM provider
print("\n1. Testing LLM Provider import...")
try:
    from src.llm.ollama_provider import OllamaProvider
    print("   [OK] OllamaProvider imported")
except Exception as e:
    print(f"   [FAIL] {e}")
    sys.exit(1)

# Test 2: Import orchestrators
print("\n2. Testing Orchestrator imports...")
try:
    from src.agents.orchestrator import Orchestrator
    print("   [OK] Orchestrator (v1) imported")
except Exception as e:
    print(f"   [FAIL] Orchestrator v1: {e}")

try:
    from src.orchestrator_v2 import OrchestratorV2
    print("   [OK] OrchestratorV2 (v2) imported")
except Exception as e:
    print(f"   [FAIL] Orchestrator v2: {e}")

# Test 3: Test LLM connection
print("\n3. Testing Ollama connection...")
try:
    llm = OllamaProvider(model="qwen2.5-coder:3b", log_interactions=False)
    print("   [OK] LLM provider initialized")
    
    # Test a simple generation
    print("   Testing simple LLM call...")
    response = llm.generate("Say 'Hello, AutoML!' in one sentence.", max_tokens=50)
    print(f"   [OK] LLM response received: {response[:100]}...")
except Exception as e:
    print(f"   [FAIL] LLM test failed: {e}")

# Test 4: Test dataset creation
print("\n4. Testing dataset creation...")
try:
    import pandas as pd
    import numpy as np
    
    np.random.seed(42)
    df = pd.DataFrame({
        'feature1': np.random.normal(50, 15, 100),
        'feature2': np.random.normal(30, 10, 100),
        'target': np.random.choice([0, 1], 100)
    })
    
    dataset_path = "data/test_simple.csv"
    Path(dataset_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dataset_path, index=False)
    print(f"   [OK] Test dataset created: {dataset_path}")
except Exception as e:
    print(f"   [FAIL] Dataset creation failed: {e}")

# Test 5: Test intent parsing (if LLM works)
print("\n5. Testing Intent Parsing...")
try:
    from src.agents.intent_agent import IntentAgent
    
    intent_agent = IntentAgent(llm)
    user_input = "Predict target. Catching positives is priority."
    
    print("   Attempting intent parsing...")
    intent = intent_agent.parse_with_validation(user_input, dataset_path)
    print(f"   [OK] Intent parsed successfully!")
    print(f"   Task type: {intent.get('task_type', 'N/A')}")
    print(f"   Target: {intent.get('target_variable', 'N/A')}")
except Exception as e:
    print(f"   [WARN] Intent parsing failed (may need better prompt): {e}")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
