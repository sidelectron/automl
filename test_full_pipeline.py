"""Full pipeline test - tests actual execution end-to-end."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

print("="*80)
print("FULL PIPELINE TEST")
print("="*80)

# Create test dataset
print("\n[1] Creating test dataset...")
import pandas as pd
import numpy as np

np.random.seed(42)
n_samples = 200

# Create a dataset with clear patterns
feature1 = np.random.normal(50, 15, n_samples)
feature2 = np.random.normal(30, 10, n_samples)
feature3 = np.random.choice(['A', 'B', 'C'], n_samples)
feature4 = np.random.normal(100, 20, n_samples)

# Target with clear relationship to features
target = ((feature1 > 50) & (feature2 > 30)).astype(int)

df = pd.DataFrame({
    'feature1': feature1,
    'feature2': feature2,
    'feature3': feature3,
    'feature4': feature4,
    'target': target
})

# Add some missing values
missing_indices = np.random.choice(df.index, size=20, replace=False)
df.loc[missing_indices, 'feature1'] = np.nan

dataset_path = "data/test_full_pipeline.csv"
Path(dataset_path).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(dataset_path, index=False)
print(f"   Dataset: {dataset_path}")
print(f"   Shape: {df.shape}")
print(f"   Target distribution: {df['target'].value_counts().to_dict()}")
print(f"   Missing values: {df.isnull().sum().sum()}")

# Test Hybrid Version Full Pipeline
print("\n" + "="*80)
print("HYBRID VERSION - FULL PIPELINE TEST")
print("="*80)

try:
    from src.llm.ollama_provider import OllamaProvider
    from src.version_store import VersionStore
    from src.agents.orchestrator import Orchestrator

    # Initialize
    print("\n[2] Initializing...")
    llm = OllamaProvider(model="qwen2.5-coder:3b", log_interactions=True)
    version_store = VersionStore(db_path="data/test_experiments.db")
    orchestrator = Orchestrator(llm, version_store)
    print("   [OK] Orchestrator initialized")

    # Run pipeline step by step
    user_input = """
    Predict target variable.
    Catching positive cases is the priority (optimize for recall).
    Each true positive is worth $500, each false positive costs $20.
    """

    print("\n[3] Parsing Intent...")
    intent = orchestrator.parse_intent(user_input, dataset_path)
    print(f"   Task type: {intent.get('task_type')}")
    print(f"   Target: {intent.get('target_variable')}")
    print(f"   Priority metric: {intent.get('business_context', {}).get('priority_metric')}")
    print(f"   TP Value: ${intent.get('business_context', {}).get('true_positive_value', 0)}")
    print(f"   FP Cost: ${intent.get('business_context', {}).get('false_positive_cost', 0)}")

    print("\n[4] Profiling Data...")
    profile = orchestrator.profile_data(dataset_path)
    print(f"   Columns: {len(profile.get('data_types', {}))}")
    print(f"   Missing values: {sum(profile.get('missing_values', {}).values())}")
    if profile.get('class_distribution'):
        print(f"   Class distribution: {profile.get('class_distribution')}")
    if profile.get('intent_flags'):
        print(f"   Intent flags: {len(profile.get('intent_flags'))} flags")
        for flag in profile.get('intent_flags', [])[:3]:
            print(f"      - [{flag.get('severity')}] {flag.get('message')[:60]}...")

    print("\n[5] Generating EDA...")
    eda = orchestrator.generate_eda()
    print(f"   Visualizations: {len(eda.get('visualizations', []))}")
    print(f"   Insights: {len(eda.get('insights', []))}")
    for insight in eda.get('insights', [])[:3]:
        print(f"      - {insight[:70]}...")

    print("\n[6] Proposing Strategies...")
    strategies = orchestrator.propose_strategies()
    print(f"   Strategies generated: {len(strategies)}")
    for s in strategies:
        print(f"      - {s.get('name')}: {len(s.get('preprocessing_steps', []))} steps, {len(s.get('model_candidates', []))} models")

    print("\n[7] Training Models...")
    print("   (This may take a while...)")
    try:
        results = orchestrator.train_models()
        print(f"   Results: {len(results)} model-strategy combinations")
        for r in results[:3]:
            metrics = r.get('metrics', {})
            bm = r.get('business_metrics', {})
            print(f"      - {r.get('strategy_name')}/{r.get('model_name')}: "
                  f"F1={metrics.get('f1', 0):.3f}, "
                  f"Recall={metrics.get('recall', 0):.3f}, "
                  f"Net Value=${bm.get('net_value', 0):,.0f}")
    except Exception as e:
        print(f"   [ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        results = []

    if results:
        print("\n[8] Comparing Results...")
        comparison = orchestrator.compare_results()
        winner = comparison.get('winner', {})
        print(f"   Winner: {winner.get('strategy_name')}/{winner.get('model_name')}")
        print(f"   Threshold: {winner.get('threshold')}")
        print(f"   Metrics: {winner.get('metrics', {})}")

        business_impact = comparison.get('business_impact', {})
        print(f"   Business Impact:")
        print(f"      - True Positives: {business_impact.get('true_positives', 'N/A')}")
        print(f"      - False Positives: {business_impact.get('false_positives', 'N/A')}")

        financial = comparison.get('financial_impact', {})
        print(f"   Financial Impact:")
        print(f"      - Net Value: ${financial.get('net_value', 0):,.0f}")
        print(f"      - ROI: {financial.get('roi', 0):.1f}x")

        print(f"\n   Comparison Text:")
        print(f"   {comparison.get('comparison_text', 'N/A')[:200]}...")

        print("\n[PASS] Hybrid Version Full Pipeline Complete!")
    else:
        print("\n[PARTIAL] Hybrid Version - Training step needs debugging")

except Exception as e:
    print(f"\n[FAIL] Hybrid Version failed: {e}")
    import traceback
    traceback.print_exc()

# Test Dynamic Version
print("\n" + "="*80)
print("DYNAMIC VERSION - CODE GENERATION TEST")
print("="*80)

try:
    from src.orchestrator_v2 import OrchestratorV2

    print("\n[9] Initializing Dynamic Version...")
    llm2 = OllamaProvider(model="qwen2.5-coder:3b", log_interactions=True)
    version_store2 = VersionStore(db_path="data/test_experiments_v2.db")
    orchestrator_v2 = OrchestratorV2(llm2, version_store2)
    print("   [OK] OrchestratorV2 initialized")

    print("\n[10] Parsing Intent (v2)...")
    intent_v2 = orchestrator_v2.parse_intent(user_input, dataset_path)
    print(f"   Task type: {intent_v2.get('task_type')}")

    print("\n[11] Profiling Data (v2)...")
    profile_v2 = orchestrator_v2.profile_data(dataset_path)
    print(f"   Columns: {len(profile_v2.get('data_types', {}))}")

    print("\n[12] Generating Plans...")
    orchestrator_v2.eda_results = {}  # Skip EDA for speed
    plans = orchestrator_v2.generate_plans()
    print(f"   Preprocessing plan: {len(plans.get('preprocessing_plan', ''))} chars")
    print(f"   Modeling plan: {len(plans.get('modeling_plan', ''))} chars")

    print("\n[13] Generating Code...")
    code = orchestrator_v2.generate_code()
    print(f"   Preprocessing code: {len(code.get('preprocessing', ''))} chars")
    print(f"   Training code: {len(code.get('training', ''))} chars")
    print(f"   Prediction code: {len(code.get('prediction', ''))} chars")

    # Show a snippet of generated code
    print("\n   --- Sample of Generated Training Code ---")
    training_code = code.get('training', '')
    lines = training_code.split('\n')[:20]
    for line in lines:
        print(f"   {line}")
    print("   ...")

    print("\n[14] Validating Generated Code...")
    from src.engine_v2.code_validator import CodeValidator
    validator = CodeValidator()

    for code_type, code_content in code.items():
        is_valid, errors = validator.validate(code_content)
        status = "VALID" if is_valid else f"INVALID ({len(errors)} errors)"
        print(f"   {code_type}: {status}")
        if not is_valid:
            for err in errors[:2]:
                print(f"      - {err}")

    print("\n[PASS] Dynamic Version Code Generation Complete!")

except Exception as e:
    print(f"\n[FAIL] Dynamic Version failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
