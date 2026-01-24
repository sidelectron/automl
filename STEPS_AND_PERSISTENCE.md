# Step-by-Step Flow: What Each Step Does & What Gets Saved

## V1 (Hybrid) – Steps

| Step | What it does | Output | Saved? | Where |
|------|--------------|--------|--------|-------|
| 1. parse_intent | LLM parses task + target from user text + dataset path | `intent` dict | **Only if** `run_full_pipeline` is used | `version_store` → `experiments` (intent in `intent_json`) |
| 2. profile_data | Profiler analyzes CSV: dtypes, missing, etc. | `profile` dict | **No** | In memory only |
| 3. generate_eda | LLM + optional plots for target-focused EDA | `eda_results` dict | **No** | In memory only |
| 4. propose_strategies | LLM (or fallback) proposes preprocessing + model strategies | `strategies` list | **No** (until train) | In memory until trainer saves |
| 5. train_models | Executor runs strategies → trains models, tunes thresholds | `training_results` list | **Yes** (partially) | • `data/models/*.pkl` (model + preprocessing) always<br>• `version_store` strategies/results **only if** `experiment_id` is set |
| 6. compare_results | Comparator picks winner, business explanation | `comparison` dict | **Only if** `run_full_pipeline` used | `version_store` → `comparisons` |
| 7. generate_project | ProjectGenerator writes project to disk | Project path | **Yes** | `data/generated_projects/<target>/` |

**When you run steps individually (e.g. test_real_datasets):**  
`save_experiment` is never called → no `experiment_id` → strategies/results are **not** written to `version_store`. Only PKL files and generated project are persisted.

---

## V2 (Dynamic) – Steps

| Step | What it does | Output | Saved? | Where |
|------|--------------|--------|--------|-------|
| 1. parse_intent | Same as V1 | `intent` dict | **Yes** (in `run_full_pipeline`) | `version_store` → `experiments` |
| 2. profile_data | Same as V1 | `profile` dict | **No** | In memory only |
| 3. generate_eda | Same as V1 | `eda_results` dict | **No** | In memory only |
| 4. generate_plans | StrategyV2 + ModelV2 → text preprocessing & modeling plans | `preprocessing_plan`, `modeling_plan` | **No** | In memory only |
| 5. generate_code | CodeGen agent writes preprocessing, training, prediction code | `generated_code` dict | **Yes** | `version_store` → `generated_code` |
| 6. execute_code | DynamicExecutor runs preprocessing + training (with fix/retry) | Execution result per type | **Yes** | `version_store` → `executions` |
| 7. verify_execution | Checks preprocessing + training both succeeded | bool | **No** | In memory only |
| 8. compare_results | Builds comparison from execution results | `comparison` dict | **Yes** | `version_store` → `comparisons` |

### V2 execution details

- **work_dir:** Code runs with `cwd = work_dir`. Pass `work_dir` to `run_full_pipeline` (e.g. project root). If omitted, it defaults to the **directory containing the dataset** (parent of `dataset_path`), which works for `archive/`, `data/`, or project-root files. Prefer passing project root explicitly when you want to run in project root.
- **Dataset path:** Resolved to absolute and stored in `intent["dataset_path"]`. Injected into CodeGenerationAgent prompts (preprocessing + training). Generated scripts must load the CSV from that path.
- **Execution:** DynamicExecutor runs preprocessing then training via `subprocess.run` with timeout (30s preprocessing, 120s training). Scripts are written under `work_dir` and executed there.
- **Compare:** If training writes `results.json` in `work_dir` (list of `{model_name, strategy_name, threshold, metrics, business_metrics}`), `compare_results` loads it, calls `ComparatorAgent.compare(...)`, and persists the comparison. Otherwise a placeholder comparison is used.

---

## Commands to Run Each Step and Check Persistence

Use same setup for all: Titanic path, shared LLM, `VersionStore` (default DB: `data/experiments/version_store.db`).

### Setup (run once)

```powershell
cd d:\LinkedIn\Week2
```

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').resolve()))
sys.path.insert(0, str(Path('.').resolve() / 'src'))

from src.llm.ollama_provider import OllamaProvider
from src.version_store import VersionStore
from src.agents.orchestrator import Orchestrator
from src.orchestrator_v2 import OrchestratorV2

USER_INPUT = "Titanic survival analysis. Predict survived."
DATASET_PATH = "archive/Titanic Dataset.csv"

llm = OllamaProvider(model="qwen2.5-coder:3b")
store = VersionStore()
orch_v1 = Orchestrator(llm, store)
orch_v2 = OrchestratorV2(llm, store)
```

### V1 – Run steps one by one

```python
# 1. Parse intent
intent = orch_v1.parse_intent(USER_INPUT, DATASET_PATH)
print("Intent:", intent.get("task_type"), intent.get("target_variable"), "dataset_path" in intent)
# Saved? No (step-by-step). Check DB: no new experiment yet.
```

```python
# 2. Profile
orch_v1.profile = orch_v1.profile_data(DATASET_PATH)
print("Profile columns:", len(orch_v1.profile.get("data_types", {})))
# Saved? No.
```

```python
# 3. EDA
orch_v1.eda_results = orch_v1.generate_eda()
print("EDA keys:", list(orch_v1.eda_results.keys()), "visualizations:", len(orch_v1.eda_results.get("visualizations", [])))
# Saved? No.
```

```python
# 4. Strategies
orch_v1.strategies = orch_v1.propose_strategies()
print("Strategies:", len(orch_v1.strategies), [s.get("name") for s in orch_v1.strategies])
# Saved? No.
```

```python
# 5. Train (needs experiment_id for version_store; we don't have it step-by-step)
orch_v1.experiment_id = store.save_experiment(intent, DATASET_PATH, USER_INPUT)
training = orch_v1.train_models(None)
print("Training results:", len(training))
# Saved? PKL in data/models/. Strategies/results in version_store (we just set experiment_id).
```

```python
# 6. Compare
orch_v1.comparison = orch_v1.compare_results()
print("Comparison winner:", orch_v1.comparison.get("winner"))
# Saved? No (we didn't call save_comparison). Do it manually:
if orch_v1.experiment_id:
    store.save_comparison(orch_v1.experiment_id, orch_v1.comparison)
```

```python
# 7. Generate project
path = orch_v1.generate_project()
print("Project:", path)
# Saved? Yes. data/generated_projects/...
```

### V2 – Run steps one by one

```python
# 1. Intent + create experiment (so we have experiment_id)
orch_v2.intent = orch_v2.parse_intent(USER_INPUT, DATASET_PATH)
orch_v2.experiment_id = store.save_experiment(orch_v2.intent, DATASET_PATH, USER_INPUT)
print("Intent:", orch_v2.intent.get("target_variable"))
# Saved? Yes. experiments table.
```

```python
# 2. Profile
orch_v2.profile = orch_v2.profile_data(DATASET_PATH)
# Saved? No.
```

```python
# 3. EDA
orch_v2.eda_results = orch_v2.generate_eda()
# Saved? No.
```

```python
# 4. Plans
plans = orch_v2.generate_plans()
print("Plans keys:", list(plans.keys()))
# Saved? No.
```

```python
# 5. Code
code = orch_v2.generate_code()
print("Code types:", list(code.keys()))
# Saved? Yes. generated_code table.
```

```python
# 6. Execute
r_pre = orch_v2.execute_code("preprocessing")
r_train = orch_v2.execute_code("training")
print("Preprocessing success:", r_pre.get("success"), "Training success:", r_train.get("success"))
# Saved? Yes. executions.
```

```python
# 7. Verify
ok = orch_v2.verify_execution()
print("Verify:", ok)
# Saved? No.
```

```python
# 8. Compare
orch_v2.comparison = orch_v2.compare_results()
# Saved? Yes (inside compare_results when experiment_id set). comparisons table.
```

### Check what was saved (VersionStore)

```python
import sqlite3
conn = sqlite3.connect("data/experiments/version_store.db")
cur = conn.cursor()
cur.execute("SELECT experiment_id, timestamp, dataset_path FROM experiments ORDER BY timestamp DESC LIMIT 5")
print("Experiments:", cur.fetchall())
cur.execute("SELECT strategy_id, experiment_id, strategy_name FROM strategies LIMIT 10")
print("Strategies:", cur.fetchall())
cur.execute("SELECT result_id, model_name, model_path FROM results LIMIT 10")
print("Results:", cur.fetchall())
conn.close()
```

### Check files on disk

```powershell
dir data\models\*.pkl
dir data\generated_projects
```
