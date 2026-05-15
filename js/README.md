# mathexec-sdk

JavaScript / TypeScript SDK for [MathExec](https://mathexec.com) — call your published model endpoints from code.

```bash
npm install mathexec-sdk
```

Works in Node 18+ and modern browsers. TypeScript types are bundled.

## Quick start

```ts
import { Model } from 'mathexec-sdk'

const model = await Model.load('your-model-id')
const result = await model.predict({ age: 25, income: 50000 })
console.log(result.predictions)     // [1]
console.log(result.probabilities)   // [0.92]
```

## Get an API key

1. Sign in at [mathexec.com](https://mathexec.com)
2. Open **Settings → API keys**
3. Create a key — it starts with `mx_live_…`. Copy it now; the full value is only shown once.
4. Pass it as `apiKey`, or store it in your env and read `process.env.MATHEXEC_API_KEY` yourself.

Public models don't need a key. Private models, rate-limited usage, and `listModels()` do.

## Usage

### Predict on one sample

```ts
import { Model } from 'mathexec-sdk'

const model = await Model.load('project/experiment-3', { apiKey: 'mx_live_…' })

// Named features
await model.predict({ age: 25, income: 50000 })

// Positional features
await model.predict([25, 50000])
```

### Predict in batch

```ts
const results = await model.predictBatch([
  { age: 25, income: 50000 },
  { age: 45, income: 80000 },
])
results.predictions     // [1, 0]
results.probabilities   // [0.92, 0.31]
```

### sklearn-style convenience

```ts
await model.predictLabel({ age: 25, income: 50000 })   // 'yes'
await model.predictProba({ age: 25, income: 50000 })   // 0.92
```

### Model metadata

```ts
const info = await model.info()
info.name              // "Churn classifier"
info.metrics           // { accuracy: 0.92, roc_auc: 0.94, ... }
await model.taskType()       // 'classification'
await model.classLabels()    // ['no', 'yes']
await model.positiveClass()  // 'yes'
```

### List your models

```ts
import { listModels } from 'mathexec-sdk'

const models = await listModels({ apiKey: 'mx_live_…' })
for (const m of models) {
  console.log(m.model_id, m.name)
}
```

### Self-hosted server

```ts
await Model.load('id', { baseUrl: 'http://localhost:8001', apiKey: 'mx_live_…' })
```

## Errors

```ts
import { Model, MathExecError } from 'mathexec-sdk'

try {
  const model = await Model.load('does-not-exist')
} catch (e) {
  if (e instanceof MathExecError) {
    console.log(e.message, e.statusCode)
  }
}
```

`MathExecError` is thrown for:

- `404` — model not found / not accessible
- `429` — rate limited
- `400` — bad input (e.g. wrong feature names or types)
- `401` — invalid or missing API key (for protected endpoints)

## License

MIT
