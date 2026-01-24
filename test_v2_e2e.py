"""V2-only E2E test: Dynamic pipeline (plans -> code -> execution -> compare).

Run with: python test_v2_e2e.py
Requires: Ollama running, qwen2.5-coder:3b, Titanic dataset in archive/.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

def main():
    print("[INFO] Loading modules...")
    from src.llm.ollama_provider import OllamaProvider
    from src.version_store import VersionStore
    from src.orchestrator_v2 import OrchestratorV2

    archive_path = project_root / "archive"
    titanic_path = None
    for name in ["Titanic Dataset.csv", "The Titanic dataset.csv", "titanic.csv"]:
        p = archive_path / name
        if p.exists():
            titanic_path = p
            break
    if not titanic_path:
        print(f"[ERROR] Titanic dataset not found in {archive_path}")
        sys.exit(1)

    user_input = (
        "The dataset containing information about passengers aboard the Titanic. "
        "Analyze factors that influenced survival rates."
    )

    print("[INFO] Initializing LLM and V2 Orchestrator...")
    llm = OllamaProvider(model="qwen2.5-coder:3b")
    store = VersionStore()
    orch = OrchestratorV2(llm, store)

    print("[INFO] Running V2 full pipeline...")
    results = orch.run_full_pipeline(
        user_input=user_input,
        dataset_path=str(titanic_path),
        work_dir=str(project_root)
    )

    if "error" in results:
        print(f"[FAIL] V2 pipeline error: {results['error']}")
        sys.exit(1)

    assert results.get("verification_passed") is True, "verification_passed must be True"
    assert results.get("comparison") is not None, "comparison must be present"
    comp = results["comparison"]
    assert comp.get("winner") is not None, "comparison must have a winner"
    assert results.get("generated_code"), "generated_code must be present"

    print("[OK] verification_passed: True")
    print("[OK] comparison.winner: present")
    print("[OK] generated_code: present")
    print("[SUCCESS] V2 E2E test passed.")

if __name__ == "__main__":
    main()
