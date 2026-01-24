"""Focused test for both versions - avoids scipy memory issues."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

print("="*80)
print("FOCUSED TEST - BOTH VERSIONS")
print("="*80)

# Test dataset
print("\n[1] Creating test dataset...")
import pandas as pd
import numpy as np

np.random.seed(42)
df = pd.DataFrame({
    'feature1': np.random.normal(50, 15, 200),
    'feature2': np.random.normal(30, 10, 200),
    'feature3': np.random.choice(['A', 'B', 'C'], 200),
    'feature4': np.random.normal(100, 20, 200),
    'target': ((np.random.normal(50, 15, 200) > 50) & (np.random.normal(30, 10, 200) > 30)).astype(int)
})
missing_indices = np.random.choice(df.index, size=20, replace=False)
df.loc[missing_indices, 'feature1'] = np.nan

dataset_path = "data/test_focused.csv"
Path(dataset_path).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(dataset_path, index=False)
print(f"   [OK] Dataset created: {dataset_path} ({df.shape[0]} rows, {df.shape[1]} cols)")

# Initialize LLM
print("\n[2] Initializing LLM...")
try:
    from src.llm.ollama_provider import OllamaProvider
    llm = OllamaProvider(model="qwen2.5-coder:3b", log_interactions=False)
    print("   [OK] LLM initialized")
except Exception as e:
    print(f"   [FAIL] LLM init failed: {e}")
    sys.exit(1)

# Test Hybrid Version
print("\n" + "="*80)
print("TESTING HYBRID VERSION (v1)")
print("="*80)

try:
    from src.version_store import VersionStore
    version_store = VersionStore(db_path=":memory:")
    
    # Import orchestrator (this might trigger scipy import)
    print("\n[3] Importing Hybrid Orchestrator...")
    from src.agents.orchestrator import Orchestrator
    orchestrator = Orchestrator(llm, version_store)
    print("   [OK] Orchestrator initialized")
    
    # Test intent parsing
    print("\n[4] Testing Intent Parsing...")
    user_input = "Predict target. Catching positives is priority. Each positive is worth $500, each false positive costs $20."
    intent = orchestrator.parse_intent(user_input, dataset_path)
    print(f"   [OK] Intent parsed: {intent.get('task_type', 'N/A')}")
    print(f"   [OK] Target: {intent.get('target_variable', 'N/A')}")
    print(f"   [OK] Priority: {intent.get('business_context', {}).get('priority_metric', 'N/A')}")
    
    # Test profiling (without sklearn imports)
    print("\n[5] Testing Data Profiling...")
    profile = orchestrator.profile_data(dataset_path)
    print(f"   [OK] Profile generated")
    print(f"   [OK] Columns: {len(profile.get('data_types', {}))}")
    
    print("\n[PASS] Hybrid Version: Core functionality works!")
    
except Exception as e:
    print(f"\n[FAIL] Hybrid Version failed: {e}")
    import traceback
    traceback.print_exc()

# Test Dynamic Version
print("\n" + "="*80)
print("TESTING DYNAMIC VERSION (v2)")
print("="*80)

try:
    from src.version_store import VersionStore
    version_store_v2 = VersionStore(db_path=":memory:")
    
    print("\n[6] Importing Dynamic Orchestrator...")
    from src.orchestrator_v2 import OrchestratorV2
    orchestrator_v2 = OrchestratorV2(llm, version_store_v2)
    print("   [OK] OrchestratorV2 initialized")
    
    # Test intent parsing
    print("\n[7] Testing Intent Parsing...")
    user_input = "Predict target. Catching positives is priority. Each positive is worth $500, each false positive costs $20."
    intent_v2 = orchestrator_v2.parse_intent(user_input, dataset_path)
    print(f"   [OK] Intent parsed: {intent_v2.get('task_type', 'N/A')}")
    print(f"   [OK] Target: {intent_v2.get('target_variable', 'N/A')}")
    
    # Test profiling
    print("\n[8] Testing Data Profiling...")
    profile_v2 = orchestrator_v2.profile_data(dataset_path)
    print(f"   [OK] Profile generated")
    print(f"   [OK] Columns: {len(profile_v2.get('data_types', {}))}")
    
    # Test plan generation
    print("\n[9] Testing Plan Generation...")
    orchestrator_v2.eda_results = {}  # Mock EDA for now
    plans = orchestrator_v2.generate_plans()
    print(f"   [OK] Preprocessing plan: {len(plans.get('preprocessing_plan', ''))} chars")
    print(f"   [OK] Modeling plan: {len(plans.get('modeling_plan', ''))} chars")
    
    # Test code generation
    print("\n[10] Testing Code Generation...")
    code = orchestrator_v2.generate_code()
    print(f"   [OK] Preprocessing code: {len(code.get('preprocessing', ''))} chars")
    print(f"   [OK] Training code: {len(code.get('training', ''))} chars")
    print(f"   [OK] Prediction code: {len(code.get('prediction', ''))} chars")
    
    print("\n[PASS] Dynamic Version: Core functionality works!")
    
except Exception as e:
    print(f"\n[FAIL] Dynamic Version failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("TEST SUMMARY")
print("="*80)
print("Both versions have been tested for core functionality.")
print("Note: Full pipeline tests may require sklearn/scipy which has")
print("compatibility issues with Python 3.13. Consider using Python 3.11 or 3.12")
print("for full end-to-end testing.")
print("="*80)
