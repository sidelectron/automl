# AutoML-Agent Architecture Learnings

**Date**: 2026-01-23
**Repository**: [DeepAuto-AI/automl-agent](https://github.com/DeepAuto-AI/automl-agent)
**License**: CC BY-NC 4.0 (Non-Commercial)
**Publication**: ICML 2025

---

## Executive Summary

AutoML-Agent is a multi-agent LLM framework for full-pipeline AutoML published at ICML 2025. Key insights:
- **5 specialized agents** coordinated by Agent Manager
- **Multi-stage verification** (Request → Execution → Implementation)
- **GPU-based LLM** (Mixtral-8x7B via vLLM)
- **Parallel plan execution** using Python multiprocessing
- **Code generation** approach (generates Python code directly)
- **Template-based** pipeline (fills in template code)

---

## 1. Agent Coordination Pattern (Agent Manager)

**File**: `agent_manager/__init__.py`

### State Machine Architecture

```
INIT → PLAN → ACT → PRE_EXEC → EXEC → POST_EXEC → REV → RES/END
```

**States**:
1. **INIT**: Parse user requirements into JSON using Prompt Agent
2. **PLAN**: Generate N plans (default 3) using RAP (Retrieval-Augmented Planning)
3. **ACT**: Execute all plans in parallel (Data Agent + Model Agent)
4. **PRE_EXEC**: Verify proposed solutions before code execution
5. **EXEC**: Generate and run Python code via Operation Agent
6. **POST_EXEC**: Verify implementation results
7. **REV**: Revise plans if verification fails
8. **END**: Return final solution

### Key Design Patterns

**1. Sequential Orchestration**
```python
class AgentManager:
    def __init__(self, ...):
        self.state = "INIT"  # Current state
        self.plans = []      # Generated plans
        self.action_results = []  # Execution results
```

**2. Parallel Execution**
```python
# Execute multiple plans concurrently
with Pool(self.n_plans) as pool:
    self.action_results = pool.map(self.execute_plan, self.plans)
```

**3. Retry Logic**
```python
retry = 0
while retry < 5:
    try:
        response = get_client(self.llm).chat.completions.create(...)
        break
    except Exception as e:
        print_message("system", e)
        retry += 1
```

**4. Multi-Stage Verification** ⭐ CRITICAL PATTERN TO ADOPT
- **Request Verification**: Check if user requirements are sufficient
- **Execution Verification**: Validate proposed solutions before code execution
- **Implementation Verification**: Verify final code and results

### Agent Profile (System Prompt)

```python
agent_profile = """You are an experienced senior project manager of an automated machine learning project (AutoML).
You have two main responsibilities:
1. Receive requirements and/or inquiries from users through a well-structured JSON object.
2. Using recent knowledge and state-of-the-art studies to devise promising high-quality plans for data scientists, machine learning research engineers, and MLOps engineers in your team."""
```

**Takeaway**: Clear role definition with specific responsibilities.

---

## 2. Intent Parsing (Prompt Agent)

**File**: `prompt_agent/__init__.py`

### Approach

**JSON Schema-Driven Parsing**:
1. Define JSON schema with required fields (task, dataset, metrics, etc.)
2. LLM parses natural language → structured JSON
3. Validate against schema

### System Prompt Structure

```python
agent_profile = f"""You are an assistant project manager in the AutoML development team.
Your task is to parse the user's requirement into a valid JSON format using the JSON specification schema as your reference.
Your response must exactly follow the given JSON schema and be based only on the user's instruction.

#JSON SPECIFICATION SCHEMA#
```json
{json_specification}
```

Your response must begin with "```json" or "{{" and end with "```" or "}}", respectively.
"""
```

**Key Components**:
1. **Role definition**: "assistant project manager"
2. **Task description**: "parse user's requirement into JSON"
3. **Schema reference**: Embedded JSON schema
4. **Output format**: Explicit format requirements
5. **Validation**: Response must start/end with specific markers

### Parsing Function

```python
def parse(self, instruction, return_json=False):
    prompt = f"""Please carefully parse the following #Instruction#.

    #Instruction#
    {instruction}

    #Valid JSON Response#
    """
    res = self.client.chat.completions.create(
        model=self.model,
        messages=[{"role": "user", "content": agent_profile + "\n" + prompt}],
        temperature=0.01  # Low temperature for consistency
    )
```

**Takeaway**: Low temperature (0.01) for deterministic parsing, embedded schema in system prompt.

---

## 3. Prompt Engineering Best Practices (Prompt Pool)

**File**: `prompt_pool/tabular_classification.py`

### Template Code Approach

They provide **skeleton code** that Operation Agent fills in:

```python
def preprocess_data():
    # TODO: this function is for data preprocessing and feature engineering
    return processed_data

def train_model(model, train_loader):
    # TODO: this function is for model training loop
    return model

def evaluate_model(model, test_loader):
    # TODO: this function is for evaluating trained model
    performance_scores = {
        'ACC': accuracy_score(y_true, y_pred),
        'F1': f1_score(y_true, y_pred)
    }
    return performance_scores, complexity_scores
```

### Few-Shot Example Structure

**Plan Conditions** (guidance for planning):
```python
plan_conditions = """
- Ensure that your plan is up-to-date with current state-of-the-art knowledge.
- Ensure that your plan is based on the requirements and objectives described in the above JSON object.
- Ensure that your plan is designed for AI agents instead of human experts.
- Ensure that your plan is self-contained with sufficient instructions to be executed by the AI agents.
- Ensure that your plan includes all the key points and instructions (from handling data to modeling).
- Ensure that your plan completely includes the end-to-end process of ML pipeline in detail.
"""
```

**Takeaway**: Explicit instructions for LLM behavior, template-based approach for code generation.

---

## 4. LLM Configuration Pattern

**File**: `configs.py`

### Centralized Config

```python
AVAILABLE_LLMs = {
    "prompt-llm": {
        "api_key": "empty",
        "model": "prompt-llama",
        "base_url": "http://localhost:8000/v1",
    },
    "gpt-4.1": {"api_key": Configs.OPENAI_KEY, "model": "gpt-4.1"},
    "gpt-4": {"api_key": Configs.OPENAI_KEY, "model": "gpt-4o"},
}
```

**Task-Specific Metrics**:
```python
TASK_METRICS = {
    "tabular_classification": "F1",
    "tabular_regression": "RMSLE",
    "image_classification": "accuracy",
}
```

**Takeaway**:
- Centralized LLM configuration with swappable backends
- Task-specific default metrics
- Clean separation of config from code

---

## 5. Retrieval-Augmented Planning (RAP)

**File**: `agent_manager/retriever.py` (referenced but not fully explored)

### Concept

Before planning, retrieve relevant knowledge/experience from:
- Internal sources (past successful plans)
- External sources (papers, documentation)

```python
if self.rap:
    self.plan_knowledge = retrieve_knowledge(
        self.user_requirements,
        self.req_summary,
        llm=self.llm
    )

plan_prompt = f"""
Here is a list of past experience cases and knowledge:
{self.plan_knowledge}

When devising a plan, follow these instructions...
"""
```

**Takeaway**: Context-aware planning using retrieved knowledge improves plan quality.

---

## 6. Multi-Agent Execution Flow

### Data Agent → Model Agent → Operation Agent

**1. Data Agent**:
- Analyzes data requirements
- Proposes preprocessing strategies
- Returns data manipulation plan

**2. Model Agent**:
- Receives data plan
- Proposes model architectures
- Suggests hyperparameters
- Returns modeling plan

**3. Operation Agent**:
- Receives combined plan
- Generates Python code
- Executes code in sandbox
- Returns implementation results

### Parallel Strategy Execution

```python
def execute_plan(self, plan):
    pid = current_process()._identity[0]  # Track process

    # Data Agent
    data_result = data_llama.execute_plan(plan, self.data_path, pid)

    # Model Agent (uses data_result)
    model_result = model_llama.execute_plan(
        k=self.n_candidates,
        project_plan=plan,
        data_result=data_result,
        pid=pid
    )

    return {"data": data_result, "model": model_result}
```

**Takeaway**: Sequential within each plan, parallel across plans.

---

## 7. Verification Strategy

### Three-Stage Verification ⭐ CRITICAL TO ADOPT

**Stage 1: Request Verification** (Pre-Planning)
```python
def _is_enough(self, msg):
    prompt = f"""Given the following JSON object, do we have essential information
    (e.g., problem and dataset) for an AutoML project?

    ```json
    {msg}
    ```

    Answer: 'yes' or 'no'; your reasons"""

    # Returns: is_enough (bool), reason (str)
```

**Stage 2: Execution Verification** (Pre-Code)
```python
def verify_solution(self, solution):
    prompt = """Given the proposed solution and user's requirements,
    please verify whether the solution 'pass' or 'fail'.

    **Proposed Solution**
    Data: {solution["data"]}
    Model: {solution["model"]}

    **User Requirements**
    {self.user_requirements}

    Answer only 'Pass' or 'Fail'"""

    is_pass = "pass" in ans.lower()
    return is_pass
```

**Stage 3: Implementation Verification** (Post-Code)
```python
verification_prompt = f"""Verify whether the given Python code and results satisfy the user's requirements.

- Python Code: {self.implementation_result['code']}
- Execution Result: {self.implementation_result['action_result']}
- User Requirements: {self.user_requirements}

Answer only 'Pass' or 'Fail'"""
```

**Takeaway**: Multi-stage verification prevents wasted computation and ensures alignment.

---

## 8. Error Handling and Revision

### Plan Revision on Failure

```python
if not is_pass and self.n_revise > 0:
    # Reflect on failure
    fail_prompt = """I found that all plans failed. Find reasons 'why' and 'how'
    the plans were unsatisfied by comparing them with requirements."""

    fail_rationale = self.generate_reply(fail_prompt)

    # Generate new plans with failure context
    plan_prompt = f"""Revise plans according to requirements.

    Use the following findings to avoid same failure:
    {fail_rationale}
    """

    self.n_revise -= 1
    self.state = "PLAN"  # Return to planning
```

**Takeaway**: Explicit reflection on failures improves subsequent attempts.

---

## 9. Key Differences: AutoML-Agent vs Our System

| Aspect | AutoML-Agent | Our System |
|--------|-------------|-----------|
| **LLM** | Mixtral-8x7B (56B) | Qwen2.5-Coder-3B |
| **Hardware** | 4+ GPUs | CPU-only |
| **Inference** | vLLM (GPU) | Ollama (CPU) |
| **Approach** | Code generation | JSON plans → executor |
| **License** | CC BY-NC 4.0 | Open-source |
| **Intent** | Task-driven | Business context-driven |
| **Threshold** | Fixed 0.5 | Tunable + optimized |
| **Output** | Python script | Complete project |
| **UI** | Notebook | Streamlit 6-page |
| **EDA** | Generic | Target-focused |
| **Metrics** | Technical only | Business translation |

---

## 10. What We're Adopting ✅

### Architecture Patterns
1. **Multi-agent coordination** (Orchestrator manages specialized agents)
2. **State machine** for workflow management
3. **Multi-stage verification** (Request → Execution → Implementation)
4. **Parallel execution** of strategies
5. **Retry logic** with exponential backoff
6. **JSON schema validation** for structured outputs

### Prompt Engineering
1. **System prompts** with clear role definition
2. **Few-shot examples** for consistency
3. **Low temperature** (0.01-0.1) for deterministic tasks
4. **Explicit output format** requirements
5. **Chain-of-thought** for complex reasoning

### Configuration
1. **Centralized LLM config** (swappable backends)
2. **Task-specific metrics** mapping
3. **Environment-based** API key management

---

## 11. What We're Building Differently 🚀

### Our Unique Value

1. **Intent-Driven Architecture**
   - Capture business context (cost_ratio, value metrics)
   - Propagate intent through ALL agents
   - Optimize for business goals, not just metrics

2. **CPU-Optimized Stack**
   - Qwen2.5-Coder-3B via Ollama (Q4 quantization)
   - 10-20 tok/s on CPU (vs their GPU requirement)
   - 100% local operation

3. **JSON Plans + Deterministic Executor**
   - Agents generate JSON (not code)
   - Deterministic executor runs plans
   - Safer, more predictable, easier to debug

4. **Threshold Tuning** ⭐ CORE FEATURE
   - Try [0.3, 0.4, 0.5, 0.6, 0.7]
   - Recommend based on user's metric
   - Interactive post-training adjustment

5. **Business Translation**
   - "Catches 1,540 of 1,900 churners"
   - "Net value: $723,500 (ROI: 16.6x)"
   - Explain WHY this beats alternatives

6. **Target-Focused EDA**
   - Show "Churn rate by Contract Type"
   - NOT "Contract Type counts"
   - All insights framed around target

7. **Complete Project Output**
   - Modular Python folder structure
   - Runnable predict.py for new data
   - NOT just notebook

8. **Streamlit UI**
   - 6-page interface for non-technical users
   - Interactive threshold adjustment
   - Live business impact calculation

---

## 12. Implementation Recommendations

### High Priority (Week 1)

1. **Adopt their orchestrator pattern** with state machine
2. **Use multi-stage verification** (critical for quality)
3. **Implement retry logic** with validation feedback
4. **Centralize LLM config** for swappable backends

### Medium Priority (Week 2-3)

5. **Parallel strategy execution** (asyncio or multiprocessing)
6. **Plan revision logic** on verification failure
7. **JSON schema validation** for all agent outputs

### Low Priority (Week 4+)

8. **RAP integration** (retrieve knowledge for planning)
9. **Fine-tuning data collection** (log all interactions)
10. **Cost tracking** (token usage per agent)

---

## 13. Code Snippets to Reference

### Retry with Validation

```python
retry = 0
while retry < 5:
    try:
        response = get_client(self.llm).chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3
        )
        break
    except Exception as e:
        print_message("system", e)
        retry += 1
        continue
```

### Parallel Execution

```python
from multiprocessing import Pool

with Pool(n_strategies) as pool:
    results = pool.map(execute_strategy, strategies)
```

### JSON Validation

```python
import json

try:
    content = json.loads(response_text)
    return content
except Exception as e:
    # Fallback: try extracting from markdown
    pattern = r"^```(?:\w+)?\s*\n(.*?)(?=^```)```"
    results = re.findall(pattern, response_text, re.DOTALL | re.MULTILINE)
    if len(results) > 0:
        content = json.loads(results[0].strip())
    return content
```

---

## 14. Citation

```bibtex
@inproceedings{automl-agent-2025,
  title={AutoML-Agent: A Multi-Agent LLM Framework for Full-Pipeline AutoML},
  author={DeepAuto-AI Team},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2025},
  note={https://github.com/DeepAuto-AI/automl-agent}
}
```

---

## 15. Final Takeaways

**What Makes AutoML-Agent Strong**:
- Multi-agent architecture is proven (ICML 2025)
- Multi-stage verification prevents errors
- Parallel execution speeds up experimentation
- Template-based approach simplifies code generation

**What We Can Improve**:
- **Business focus** (they focus on technical metrics)
- **CPU operation** (they require expensive GPUs)
- **Threshold tuning** (they use fixed 0.5)
- **Explainability** (they don't translate to business terms)
- **Complete output** (they generate scripts, not full projects)

**Our Competitive Advantage**:
> "We're not building AutoML—we're building **Intent-Driven ML Automation** that understands business goals, optimizes for real-world tradeoffs, and explains results in language anyone can understand."

---

**Study Completed**: 2026-01-23
**Next Steps**: Begin implementation following our CPU-optimized, intent-driven architecture
