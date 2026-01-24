"""Test both V1 (Hybrid) and V2 (Dynamic) AutoML platforms with Titanic dataset."""

import sys
import time
from pathlib import Path
from threading import Thread

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

print("[INFO] Loading modules (this may take a moment)...")
sys.stdout.flush()

# Lazy import to avoid memory issues
try:
    from src.llm.ollama_provider import OllamaProvider
    from src.version_store import VersionStore
except MemoryError as e:
    print("\n" + "=" * 80)
    print("MEMORY ERROR: Windows Paging File Too Small")
    print("=" * 80)
    print("\nYour system is running out of virtual memory when importing libraries.")
    print("This is a Windows system configuration issue.")
    print("\nTo fix this:")
    print("1. Open System Properties -> Advanced -> Performance Settings")
    print("2. Go to Advanced tab -> Virtual memory -> Change")
    print("3. Increase the paging file size:")
    print("   - Recommended: 8192 MB (8 GB) minimum")
    print("   - Or set to 'System managed size'")
    print("4. Click OK and restart your computer")
    print("\nAlternatively, try:")
    print("  - Closing other applications to free memory")
    print("  - Restarting your computer")
    print("=" * 80)
    sys.exit(1)
except Exception as e:
    print(f"[ERROR] Failed to import modules: {e}")
    sys.exit(1)

# Lazy imports to avoid loading heavy libraries (sklearn/scipy) at import time
# This helps with Windows paging file issues
def get_orchestrator_v1():
    """Lazy import for V1 Orchestrator."""
    from src.agents.orchestrator import Orchestrator
    return Orchestrator

def get_orchestrator_v2():
    """Lazy import for V2 Orchestrator."""
    from src.orchestrator_v2 import OrchestratorV2
    return OrchestratorV2


if __name__ == "__main__":
    print("=" * 80)
    print("TESTING BOTH AUTOML VERSIONS WITH TITANIC DATASET")
    print("=" * 80)
    print("\nNOTE: This test requires Ollama to be running with qwen2.5-coder:3b model")
    print("If Ollama is not running, start it with: ollama serve")
    print("=" * 80)
    print("\nThis test lets BOTH platforms do ALL the work:")
    print("  V1 (Hybrid): JSON strategies -> sklearn Pipeline")
    print("  V2 (Dynamic): Text plans -> LLM-generated Python code")
    print("\nBoth will:")
    print("  - Find and read the dataset")
    print("  - Profile the data")
    print("  - Generate EDA")
    print("  - Create preprocessing and modeling plans")
    print("  - Generate executable code/models")
    print("=" * 80)

    # Find Titanic dataset
    archive_path = project_root / "archive"
    titanic_path = None

    # Prefer Titanic Dataset.csv (has proper headers: pclass, survived, etc.)
    # "The Titanic dataset.csv" has no header row, columns become 1,2,3...
    possible_names = ["Titanic Dataset.csv", "The Titanic dataset.csv", "titanic.csv"]
    for name in possible_names:
        test_path = archive_path / name
        if test_path.exists():
            titanic_path = test_path
            break

    if not titanic_path:
        print(f"\n[ERROR] Titanic dataset not found in {archive_path}")
        print("Available files:")
        for f in archive_path.glob("*.csv"):
            print(f"  - {f.name}")
        sys.exit(1)

    print(f"\nDataset found: {titanic_path.name}")
    print(f"Path: {titanic_path}")

    # Initialize LLM provider (shared by both versions)
    print("\n" + "-" * 80)
    print("Initializing LLM Provider (shared by both versions)...")
    print("-" * 80)

    try:
        llm_provider = OllamaProvider(model="qwen2.5-coder:3b")
        print("[OK] LLM provider initialized")
    except Exception as e:
        print(f"[FAIL] LLM initialization: {e}")
        sys.exit(1)

    # Test LLM connectivity with timeout
    print("\n" + "-" * 80)
    print("Testing LLM Connectivity...")
    print("-" * 80)
    print("Sending test request to Ollama (this verifies it's running)...")
    print("Timeout: 150 seconds (2.5 minutes for slow CPU inference)")
    sys.stdout.flush()

    def test_ollama_connection():
        """Test if Ollama server is accessible."""
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                return {"accessible": True, "models": model_names}
            return {"accessible": False, "error": f"HTTP {response.status_code}"}
        except ImportError:
            # requests not available, skip this check
            return {"accessible": None, "error": "requests library not available"}
        except Exception as e:
            return {"accessible": False, "error": str(e)}

    def test_llm_with_timeout(timeout_seconds=150):
        """Test LLM with configurable timeout (default 150 seconds)."""
        result = {"response": None, "error": None, "completed": False}
        
        def run_test():
            try:
                test_start = time.time()
                test_response = llm_provider.generate(
                    prompt="Respond with just 'OK' if you can read this.",
                    temperature=0.1,
                    max_tokens=10
                )
                test_time = time.time() - test_start
                result["response"] = test_response
                result["time"] = test_time
                result["completed"] = True
            except Exception as e:
                result["error"] = str(e)
                result["completed"] = True
        
        thread = Thread(target=run_test)
        thread.daemon = True
        thread.start()
        
        # Show progress while waiting (for slow CPU inference)
        print("         Waiting for response...", end="", flush=True)
        for i in range(timeout_seconds):  # Check every second
            if result["completed"]:
                break
            time.sleep(1)
            if i % 10 == 0 and i > 0:
                print(f" {i}s", end="", flush=True)
        
        thread.join(timeout=timeout_seconds)  # 150 second timeout
        
        if not result["completed"]:
            return {"timeout": True, "error": f"LLM call timed out after {timeout_seconds} seconds"}
        elif result["error"]:
            return {"timeout": False, "error": result["error"]}
        else:
            return {"timeout": False, "response": result["response"], "time": result["time"]}

    try:
        test_result = test_llm_with_timeout(timeout_seconds=150)
        
        if test_result.get("timeout"):
            print(f"\n[FAIL] LLM connectivity test timed out!")
            print("\nTroubleshooting:")
            print("1. Make sure Ollama is running: 'ollama serve'")
            print("2. Make sure model is pulled: 'ollama pull qwen2.5-coder:3b'")
            print("3. Test manually: 'ollama run qwen2.5-coder:3b \"Hello\"'")
            print("4. Check if Ollama is accessible at http://localhost:11434")
            print("5. The model might be loading for the first time (this can take a minute)")
            sys.exit(1)
        elif test_result.get("error"):
            error_msg = test_result['error']
            print(f"\n[FAIL] LLM connectivity test failed: {error_msg}")
            
            if "Failed to connect" in error_msg or "connection" in error_msg.lower():
                print("\nConnection Error - Ollama server is not accessible")
                print("\nTroubleshooting:")
                print("1. Check if Ollama is running:")
                print("   - Open Task Manager and look for 'ollama' process")
                print("   - Or run: 'ollama serve' in a separate terminal")
                print("2. If Ollama was running but stopped, restart it:")
                print("   - Stop any existing Ollama processes")
                print("   - Run: 'ollama serve'")
                print("3. Test Ollama manually:")
                print("   - Run: 'ollama run qwen2.5-coder:3b \"Hello\"'")
                print("   - This will verify Ollama is working")
                print("4. Check if port 11434 is accessible:")
                print("   - Try: 'curl http://localhost:11434/api/tags'")
            else:
                print("\nTroubleshooting:")
                print("1. Make sure Ollama is running: 'ollama serve'")
                print("2. Make sure model is pulled: 'ollama pull qwen2.5-coder:3b'")
                print("3. Test manually: 'ollama run qwen2.5-coder:3b \"Hello\"'")
                print("4. Check if Ollama is accessible at http://localhost:11434")
            
            sys.exit(1)
        else:
            print(f"[OK] LLM is responding! (took {test_result['time']:.1f} seconds)")
            print(f"     Response: {test_result['response'][:50]}...")
            sys.stdout.flush()
    except Exception as e:
        print(f"\n[FAIL] LLM connectivity test error: {e}")
        sys.exit(1)

    # User intent as provided
    user_input = """The dataset containing information about passengers aboard the Titanic is one of the most famous datasets used in data science and machine learning. It was created to analyze and understand the factors that influenced survival rates among passengers during the tragic sinking of the RMS Titanic on April 15, 1912."""

    print("\n" + "=" * 80)
    print("USER INTENT")
    print("=" * 80)
    print(f"\n{user_input}")
    print(f"\nDataset: {titanic_path}")

    results_v1 = None
    results_v2 = None

    # ============================================================================
    # TEST 1: V1 HYBRID VERSION (JSON Strategies -> sklearn Pipeline)
    # ============================================================================
    print("\n" + "=" * 80)
    print("TEST 1: V1 HYBRID VERSION (JSON Strategies -> sklearn Pipeline)")
    print("=" * 80)

    try:
        print("[INFO] Loading V1 Orchestrator (this may take a moment due to sklearn/scipy imports)...")
        sys.stdout.flush()
        try:
            Orchestrator = get_orchestrator_v1()
        except ImportError as e:
            if "paging file" in str(e).lower() or "DLL load failed" in str(e):
                print("\n" + "=" * 80)
                print("SYSTEM ERROR: Windows Paging File Too Small")
                print("=" * 80)
                print("\nThe error indicates your Windows paging file (virtual memory) is too small.")
                print("This is required for scipy/sklearn to load.")
                print("\nTo fix this:")
                print("1. Open System Properties -> Advanced -> Performance Settings")
                print("2. Go to Advanced tab -> Virtual memory -> Change")
                print("3. Increase the paging file size (recommended: 4096 MB minimum)")
                print("4. Restart your computer")
                print("\nAlternatively, you can test V2 (Dynamic) version which may not require scipy.")
                print("=" * 80)
                sys.exit(1)
            else:
                raise
        version_store_v1 = VersionStore()
        orchestrator_v1 = Orchestrator(llm_provider, version_store_v1)
        print("[OK] V1 Orchestrator initialized")
        
        print("\n" + "-" * 80)
        print("Running V1 full pipeline step by step...")
        print("(Intent -> Profile -> EDA -> JSON Strategies -> sklearn Pipeline -> Training)")
        print("-" * 80)
        
        # Run step by step with progress indicators
        print("\n[Step 1/7] Parsing intent...")
        print("  (This calls Ollama LLM - may take 10-30 seconds on CPU)")
        sys.stdout.flush()  # Force output
        
        try:
            intent_v1 = orchestrator_v1.parse_intent(user_input, str(titanic_path))
            print(f"[OK] Intent parsed: {intent_v1.get('task_type')} - Target: {intent_v1.get('target_variable')}")
        except Exception as e:
            print(f"[ERROR] Intent parsing failed: {e}")
            print("\nTroubleshooting:")
            print("1. Make sure Ollama is running: 'ollama serve'")
            print("2. Make sure model is pulled: 'ollama pull qwen2.5-coder:3b'")
            print("3. Test Ollama: 'ollama run qwen2.5-coder:3b \"Hello\"'")
            raise
        sys.stdout.flush()
        
        print("\n[Step 2/7] Profiling data...")
        print("  (Analyzing dataset structure, missing values, outliers, etc.)")
        sys.stdout.flush()
        profile_v1 = orchestrator_v1.profile_data(str(titanic_path))
        print(f"[OK] Profile generated: {len(profile_v1.get('data_types', {}))} columns")
        sys.stdout.flush()
        
        print("\n[Step 3/7] Generating EDA...")
        print("  (This calls Ollama LLM - may take 10-30 seconds)")
        sys.stdout.flush()
        eda_v1 = orchestrator_v1.generate_eda()
        print(f"[OK] EDA generated")
        sys.stdout.flush()
        
        print("\n[Step 4/7] Proposing strategies...")
        sys.stdout.flush()
        strategies_v1 = orchestrator_v1.propose_strategies()
        print(f"[OK] Strategies proposed: {len(strategies_v1)}")
        sys.stdout.flush()
        
        print("\n[Step 5/7] Training models (this may take a while)...")
        sys.stdout.flush()
        training_results_v1 = orchestrator_v1.train_models(selected_strategies=None)
        print(f"[OK] Training completed: {len(training_results_v1)} results")
        sys.stdout.flush()
        
        print("\n[Step 6/7] Comparing results...")
        sys.stdout.flush()
        comparison_v1 = orchestrator_v1.compare_results()
        print(f"[OK] Comparison done")
        sys.stdout.flush()
        
        print("\n[Step 7/7] Generating project...")
        sys.stdout.flush()
        try:
            project_path_v1 = orchestrator_v1.generate_project()
            print(f"[OK] Project generated: {project_path_v1}")
        except Exception as e:
            print(f"[WARNING] Project generation failed: {e}")
            project_path_v1 = None
        sys.stdout.flush()
        
        # Build results dictionary
        results_v1 = {
            "experiment_id": orchestrator_v1.experiment_id,
            "intent": intent_v1,
            "profile": profile_v1,
            "eda_results": eda_v1,
            "strategies": strategies_v1,
            "training_results": training_results_v1,
            "comparison": comparison_v1,
            "project_path": project_path_v1
        }
        
        print("\n" + "-" * 80)
        print("V1 PIPELINE RESULTS")
        print("-" * 80)
        
        if "error" in results_v1:
            print(f"[ERROR] V1 Pipeline failed: {results_v1['error']}")
        else:
            print(f"  - Experiment ID: {results_v1.get('experiment_id', 'N/A')}")
            print(f"  - Intent parsed: {'[OK]' if results_v1.get('intent') else '[X]'}")
            print(f"  - Profile generated: {'[OK]' if results_v1.get('profile') else '[X]'}")
            print(f"  - EDA generated: {'[OK]' if results_v1.get('eda_results') else '[X]'}")
            print(f"  - Strategies proposed: {'[OK]' if results_v1.get('strategies') else '[X]'}")
            print(f"  - Training completed: {'[OK]' if results_v1.get('training_results') else '[X]'}")
            print(f"  - Comparison done: {'[OK]' if results_v1.get('comparison') else '[X]'}")
            print(f"  - Project generated: {'[OK]' if results_v1.get('project_path') else '[X]'}")
            
            if results_v1.get('strategies'):
                print(f"\n  Strategies proposed: {len(results_v1['strategies'])}")
                for i, strategy in enumerate(results_v1['strategies'][:3], 1):
                    print(f"    {i}. {strategy.get('name', 'Unknown')}")
            
            if results_v1.get('training_results'):
                print(f"\n  Training results: {len(results_v1['training_results'])} models trained")
            
            print("\n[SUCCESS] V1 Hybrid pipeline completed!")
        
    except Exception as e:
        print(f"\n[FAIL] V1 Pipeline error: {e}")
        import traceback
        traceback.print_exc()

    # ============================================================================
    # TEST 2: V2 DYNAMIC VERSION (Text Plans -> LLM-Generated Python Code)
    # ============================================================================
    print("\n" + "=" * 80)
    print("TEST 2: V2 DYNAMIC VERSION (Text Plans -> LLM-Generated Python Code)")
    print("=" * 80)

    try:
        print("[INFO] Loading V2 Orchestrator...")
        sys.stdout.flush()
        OrchestratorV2 = get_orchestrator_v2()
        version_store_v2 = VersionStore()
        orchestrator_v2 = OrchestratorV2(llm_provider, version_store_v2)
        print("[OK] V2 Orchestrator initialized")

        print("\n" + "-" * 80)
        print("Running V2 full pipeline...")
        print("(Intent -> Profile -> EDA -> Text Plans -> Python Code -> Dynamic Execution)")
        print("-" * 80)

        results_v2 = orchestrator_v2.run_full_pipeline(
            user_input=user_input,
            dataset_path=str(titanic_path),
            work_dir=str(project_root)
        )

        print("\n" + "-" * 80)
        print("V2 PIPELINE RESULTS")
        print("-" * 80)

        if results_v2 and "error" in results_v2:
            print(f"[ERROR] V2 Pipeline failed: {results_v2['error']}")
        elif results_v2:
            print(f"  - Experiment ID: {results_v2.get('experiment_id', 'N/A')}")
            print(f"  - Intent parsed: {'[OK]' if results_v2.get('intent') else '[X]'}")
            print(f"  - Profile generated: {'[OK]' if results_v2.get('profile') else '[X]'}")
            print(f"  - EDA generated: {'[OK]' if results_v2.get('eda_results') else '[X]'}")
            print(f"  - Plans generated: {'[OK]' if results_v2.get('plans') else '[X]'}")
            print(f"  - Code generated: {'[OK]' if results_v2.get('generated_code') else '[X]'}")
            print(f"  - Verification passed: {'[OK]' if results_v2.get('verification_passed') else '[X]'}")

            if results_v2.get('plans'):
                plans = results_v2['plans']
                print(f"\n  Plans:")
                print(f"    - Preprocessing plan: {len(plans.get('preprocessing_plan', ''))} chars")
                print(f"    - Modeling plan: {len(plans.get('modeling_plan', ''))} chars")

            if results_v2.get('generated_code'):
                code = results_v2['generated_code']
                print(f"\n  Generated Code:")
                for code_type, code_content in code.items():
                    print(f"    - {code_type}: {len(code_content)} chars ({len(code_content.split(chr(10)))} lines)")

            print("\n[SUCCESS] V2 Dynamic pipeline completed!")
        else:
            print("[WARNING] V2 returned no results")

    except Exception as e:
        print(f"\n[FAIL] V2 Pipeline error: {e}")
        import traceback
        traceback.print_exc()

    # ============================================================================
    # SUMMARY
    # ============================================================================
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    if results_v1 is not None and results_v1.get("error") is None:
        print("V1 Hybrid:   COMPLETED")
    elif results_v1 is not None:
        print("V1 Hybrid:   FAILED (error in results)")
    else:
        print("V1 Hybrid:   FAILED or SKIPPED")

    if results_v2 is not None and results_v2.get("error") is None:
        print("V2 Dynamic:  COMPLETED")
    elif results_v2 is not None:
        print("V2 Dynamic:  FAILED (error in results)")
    else:
        print("V2 Dynamic:  FAILED or SKIPPED")

    print("=" * 80)
    print("\nNext Steps:")
    if results_v2 is not None and results_v2.get("error") is None:
        print("  - Execute generated code using orchestrator_v2.execute_code()")
    print("  - Use Streamlit UI to interact with the platform")
    print("  - Deploy the generated code")
    print("\nTEST COMPLETED!")
    print("=" * 80)
