# mathexec

Python SDK for [MathExec](https://mathexec.com) — call your published model endpoints from code.

```bash
pip install mathexec
```

## Quick start

```python
from mathexec import Model

model = Model.load("your-model-id")
result = model.predict({"age": 25, "income": 50000})
print(result["predictions"])     # [1]
print(result["probabilities"])   # [0.92]
```

## Get an API key

1. Sign in at [mathexec.com](https://mathexec.com)
2. Open **Settings → API keys**
3. Create a key — it starts with `mx_live_…`. Copy it now; the full value is only shown once.
4. Pass it as `api_key=`, or set the `MATHEXEC_API_KEY` env var and read it yourself.

Public models don't need a key. Private models, rate-limited usage, and `list_models()` do.

## Usage

### Predict on one sample

```python
from mathexec import Model

model = Model.load("project/experiment-3", api_key="mx_live_…")

# Named features
model.predict({"age": 25, "income": 50000})

# Positional features
model.predict([25, 50000])
```

### Predict in batch

```python
results = model.predict_batch([
    {"age": 25, "income": 50000},
    {"age": 45, "income": 80000},
])
results["predictions"]    # [1, 0]
results["probabilities"]  # [0.92, 0.31]
```

### sklearn-style convenience

```python
model.predict_label({"age": 25, "income": 50000})   # 'yes'
model.predict_proba({"age": 25, "income": 50000})   # 0.92
```

### Model metadata

```python
model.info["name"]            # "Churn classifier"
model.info["metrics"]         # {"accuracy": 0.92, "roc_auc": 0.94, ...}
model.task_type               # 'classification'
model.class_labels            # ['no', 'yes']
model.positive_class          # 'yes'
```

### List your models

```python
from mathexec import list_models

models = list_models(api_key="mx_live_…")
for m in models:
    print(m["model_id"], m["name"])
```

### Self-hosted server

```python
Model.load("id", base_url="http://localhost:8001", api_key="mx_live_…")
```

## Errors

```python
from mathexec import Model, MathExecError

try:
    model = Model.load("does-not-exist")
except MathExecError as e:
    print(e)   # "Model 'does-not-exist' not found."
```

`MathExecError` is raised for:

- `404` — model not found / not accessible
- `429` — rate limited
- `400` — bad input (e.g. wrong feature names or types)

For 5xx the underlying `requests.HTTPError` propagates.

## License

MIT
