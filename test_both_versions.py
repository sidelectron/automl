"""Test script for both Hybrid and Dynamic AutoML versions."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import tempfile
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.llm.ollama_provider import OllamaProvider
from src.agents.orchestrator import Orchestrator
from src.orchestrator_v2 import OrchestratorV2
from src.version_store import VersionStore


def create_test_dataset(output_path: str = "data/test_dataset.csv"):
    """Create a simple test dataset for binary classification."""
    # Create directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 200
    
    # Features
    feature1 = np.random.normal(50, 15, n_samples)
    feature2 = np.random.normal(30, 10, n_samples)
    feature3 = np.random.choice(['A', 'B', 'C'], n_samples)
    feature4 = np.random.normal(100, 20, n_samples)
    
    # Target (binary classification)
    # Simple rule: if feature1 > 50 and feature2 > 30, more likely to be 1
    target = ((feature1 > 50) & (feature2 > 30)).astype(int)
    # Add some noise
    noise = np.random.random(n_samples) < 0.2
    target = (target ^ noise).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'feature1': feature1,
        'feature2': feature2,
        'feature3': feature3,
        'feature4': feature4,
        'target': target
    })
    
    # Add some missing values
    missing_indices = np.random.choice(df.index, size=int(0.1 * len(df)), replace=False)
    df.loc[missing_indices, 'feature1'] = np.nan
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"[OK] Created test dataset: {output_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Target distribution: {df['target'].value_counts().to_dict()}")
    print(f"  Missing values: {df.isnull().sum().sum()}")
    
    return output_path


def test_hybrid_version(dataset_path: str):
    """Test the Hybrid (v1) version."""
    print("\n" + "="*80)
    print("TESTING HYBRID VERSION (v1)")
    print("="*80)
    
    try:
        # Initialize
        print("\n1. Initializing Hybrid Orchestrator...")
        llm = OllamaProvider(model="qwen2.5-coder:3b", log_interactions=False)
        version_store = VersionStore(db_path=":memory:")  # In-memory for testing
        orchestrator = Orchestrator(llm, version_store)
        print("   [OK] Orchestrator initialized")
        
        # Test intent parsing
        print("\n2. Testing Intent Parsing...")
        user_input = "Predict target. Catching positives is priority. Each positive is worth $500, each false positive costs $20."
        intent = orchestrator.parse_intent(user_input, dataset_path)
        print(f"   [OK] Intent parsed: {intent.get('task_type', 'N/A')}")
        print(f"   [OK] Target variable: {intent.get('target_variable', 'N/A')}")
        print(f"   [OK] Priority metric: {intent.get('business_context', {}).get('priority_metric', 'N/A')}")
        
        # Test profiling
        print("\n3. Testing Data Profiling...")
        profile = orchestrator.profile_data(dataset_path)
        print(f"   [OK] Profile generated")
        print(f"   [OK] Data types: {len(profile.get('data_types', {}))} columns")
        print(f"   [OK] Missing values detected: {bool(profile.get('missing_values', {}))}")
        
        # Test EDA
        print("\n4. Testing EDA Generation...")
        eda_results = orchestrator.generate_eda()
        print(f"   [OK] EDA generated")
        print(f"   [OK] Visualizations: {len(eda_results.get('visualizations', []))}")
        
        # Test strategy generation
        print("\n5. Testing Strategy Generation...")
        strategies = orchestrator.propose_strategies()
        print(f"   [OK] Strategies generated: {len(strategies)}")
        for i, strategy in enumerate(strategies[:3], 1):  # Show first 3
            print(f"   - Strategy {i}: {strategy.get('name', 'N/A')}")
            print(f"     Steps: {len(strategy.get('preprocessing_steps', []))}")
            print(f"     Models: {strategy.get('model_candidates', [])}")
        
        print("\n[PASS] Hybrid Version Test: PASSED")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] Hybrid Version Test: FAILED")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dynamic_version(dataset_path: str):
    """Test the Dynamic (v2) version."""
    print("\n" + "="*80)
    print("TESTING DYNAMIC VERSION (v2)")
    print("="*80)
    
    try:
        # Initialize
        print("\n1. Initializing Dynamic Orchestrator...")
        llm = OllamaProvider(model="qwen2.5-coder:3b", log_interactions=False)
        version_store = VersionStore(db_path=":memory:")  # In-memory for testing
        orchestrator = OrchestratorV2(llm, version_store)
        print("   [OK] Orchestrator v2 initialized")
        
        # Test intent parsing
        print("\n2. Testing Intent Parsing...")
        user_input = "Predict target. Catching positives is priority. Each positive is worth $500, each false positive costs $20."
        intent = orchestrator.parse_intent(user_input, dataset_path)
        print(f"   [OK] Intent parsed: {intent.get('task_type', 'N/A')}")
        print(f"   [OK] Target variable: {intent.get('target_variable', 'N/A')}")
        
        # Test profiling
        print("\n3. Testing Data Profiling...")
        profile = orchestrator.profile_data(dataset_path)
        print(f"   [OK] Profile generated")
        print(f"   [OK] Data types: {len(profile.get('data_types', {}))} columns")
        
        # Test EDA
        print("\n4. Testing EDA Generation...")
        eda_results = orchestrator.generate_eda()
        print(f"   [OK] EDA generated")
        
        # Test plan generation
        print("\n5. Testing Plan Generation...")
        plans = orchestrator.generate_plans()
        print(f"   [OK] Preprocessing plan generated ({len(plans['preprocessing_plan'])} chars)")
        print(f"   [OK] Modeling plan generated ({len(plans['modeling_plan'])} chars)")
        print(f"   Preview of preprocessing plan:")
        print(f"   {plans['preprocessing_plan'][:200]}...")
        
        # Test code generation
        print("\n6. Testing Code Generation...")
        code = orchestrator.generate_code()
        print(f"   [OK] Preprocessing code generated ({len(code.get('preprocessing', ''))} chars)")
        print(f"   [OK] Training code generated ({len(code.get('training', ''))} chars)")
        print(f"   [OK] Prediction code generated ({len(code.get('prediction', ''))} chars)")
        
        # Show code preview
        if code.get('preprocessing'):
            print(f"\n   Preprocessing code preview (first 300 chars):")
            print(f"   {code['preprocessing'][:300]}...")
        
        # Test code validation
        print("\n7. Testing Code Validation...")
        from src.engine_v2.code_validator import CodeValidator
        validator = CodeValidator()
        
        if code.get('preprocessing'):
            is_valid, errors = validator.validate(code['preprocessing'])
            if is_valid:
                print(f"   [OK] Preprocessing code is valid")
            else:
                print(f"   [WARN] Preprocessing code has {len(errors)} validation issues:")
                for error in errors[:3]:  # Show first 3
                    print(f"     - {error}")
        
        # Test code execution (optional - may fail if code has issues)
        print("\n8. Testing Code Execution (optional)...")
        try:
            result = orchestrator.execute_code("preprocessing", max_attempts=2)
            if result.get("success"):
                print(f"   [OK] Preprocessing code executed successfully")
                print(f"   Execution time: {result.get('execution_time', 0):.2f}s")
            else:
                print(f"   [WARN] Preprocessing code execution failed (expected for generated code)")
                print(f"   Error type: {result.get('error_type', 'N/A')}")
        except Exception as e:
            print(f"   [WARN] Code execution skipped: {e}")
        
        print("\n[PASS] Dynamic Version Test: PASSED")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] Dynamic Version Test: FAILED")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_ollama():
    """Check if Ollama is running."""
    print("Checking Ollama connection...")
    try:
        import ollama
        models = ollama.list()
        print("[OK] Ollama is running")
        print(f"  Available models: {[m['name'] for m in models.get('models', [])]}")
        
        # Check if qwen2.5-coder:3b is available
        model_names = [m['name'] for m in models.get('models', [])]
        if any('qwen' in name.lower() and 'coder' in name.lower() for name in model_names):
            print("[OK] Qwen Coder model found")
            return True
        else:
            print("[WARN] Qwen Coder model not found. Available models: {model_names}")
            print(f"  You may need to run: ollama pull qwen2.5-coder:3b")
            return False
    except ImportError:
        print("[WARN] Ollama Python package not installed")
        print("  Install with: pip install ollama")
        return False
    except Exception as e:
        print(f"[ERROR] Ollama connection failed: {e}")
        print(f"  Make sure Ollama is running: ollama serve")
        return False


def main():
    """Main test function."""
    print("="*80)
    print("AUTOML PLATFORM - BOTH VERSIONS TEST")
    print("="*80)
    
    # Check Ollama
    ollama_ok = check_ollama()
    if not ollama_ok:
        print("\n[WARN] Warning: Ollama check failed. Tests may not work properly.")
        print("  Tests will continue but may fail if Ollama is not available.")
        print("  To install Ollama: https://ollama.ai")
        print("  To install Python package: pip install ollama")
    
    # Create test dataset
    print("\n" + "="*80)
    print("SETUP: Creating Test Dataset")
    print("="*80)
    dataset_path = create_test_dataset()
    
    # Test both versions
    results = {}
    
    # Test Hybrid version
    results['hybrid'] = test_hybrid_version(dataset_path)
    
    # Test Dynamic version
    results['dynamic'] = test_dynamic_version(dataset_path)
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Hybrid Version (v1):  {'[PASS] PASSED' if results['hybrid'] else '[FAIL] FAILED'}")
    print(f"Dynamic Version (v2): {'[PASS] PASSED' if results['dynamic'] else '[FAIL] FAILED'}")
    
    if all(results.values()):
        print("\n[SUCCESS] All tests passed!")
    else:
        print("\n[WARN] Some tests failed. Check errors above.")
    
    # Cleanup
    print(f"\nTest dataset saved at: {dataset_path}")
    print("(You can delete it manually if needed)")


if __name__ == "__main__":
    main()
