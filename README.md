# MathExec SDKs

Official client libraries for [MathExec](https://mathexec.com) — call your published model endpoints from Python or JavaScript.

| Language   | Install                      | Source                |
|------------|------------------------------|-----------------------|
| Python     | `pip install mathexec`       | [`python/`](./python) |
| JavaScript | `npm install mathexec-sdk`   | [`js/`](./js)         |

Both SDKs target the same backend (`https://api.mathexec.com`) and stay feature-equivalent — see [`surface.yaml`](./surface.yaml) for the contract and [`scripts/parity_check.py`](./scripts/parity_check.py) for the CI gate.

## Quick start

```python
# Python
from mathexec import Model
model = Model.load("your-model-id")
model.predict({"age": 25, "income": 50000})
```

```ts
// JavaScript / TypeScript
import { Model } from 'mathexec-sdk'
const model = await Model.load('your-model-id')
await model.predict({ age: 25, income: 50000 })
```

Full docs in each package's README: [Python](./python/README.md) · [JavaScript](./js/README.md).

## Repository layout

```
.
├── python/           ← Python SDK (pyproject.toml, sources, tests)
├── js/               ← JavaScript SDK (package.json, src/, dist/)
├── surface.yaml      ← API contract — single source of truth
├── scripts/
│   └── parity_check.py   ← CI script that asserts both SDKs match surface.yaml
├── smoke_test.py     ← End-to-end smoke test (Python)
└── smoke_test.mjs    ← End-to-end smoke test (Node)
```

Each language directory is a self-contained, independently publishable package. `npm install mathexec-sdk` ships only `js/dist/`; `pip install mathexec` ships only `python/mathexec/`.

## Versioning

Both packages release in lockstep — a `v0.1.0` tag publishes both to PyPI and npm at version `0.1.0`. Keeps the install story symmetric and the parity check honest.

## Contributing

1. Make the change in **both** `python/` and `js/`.
2. Update `surface.yaml` if you added, renamed, or removed a method.
3. Run `python scripts/parity_check.py` — it must pass.
4. Run the smoke tests against a local backend if your change touches the wire.

## License

MIT — see [LICENSE](./LICENSE).
