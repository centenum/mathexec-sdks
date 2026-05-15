// Smoke test: hit a published model with the JS SDK.
//
// Usage:
//   cd sdk/js && npm install && npm run build
//   cd ../..
//   MODEL_ID=abc12345 BASE_URL=http://localhost:8001 node sdk/smoke_test.mjs
//
// Set API_KEY=mx_live_... if the model is private/unlisted.
//
// Imports the freshly built SDK from ./js/dist directly so this works
// before publishing to npm.

import { Model, MathExecError } from "./js/dist/index.mjs";

async function main() {
  const modelId = process.env.MODEL_ID;
  const baseUrl = process.env.BASE_URL || "http://localhost:8001";
  const apiKey = process.env.API_KEY;

  if (!modelId) {
    console.error("Set MODEL_ID=<your published model id>");
    process.exit(2);
  }

  console.log(`Loading ${modelId} from ${baseUrl}...`);
  let m;
  try {
    m = await Model.load(modelId, { baseUrl, apiKey });
  } catch (e) {
    if (e instanceof MathExecError) {
      console.error(`Load failed: ${e.message}`);
      process.exit(1);
    }
    throw e;
  }

  const info = await m.info();
  const featureNames = info.feature_names || [];
  const nFeatures = info.n_features || featureNames.length;
  const taskType = await m.taskType();

  console.log(`  name: ${info.name}`);
  console.log(`  task: ${taskType}, features: ${nFeatures}`);
  if (featureNames.length > 0) {
    const head = featureNames.slice(0, 5);
    console.log(`  feature_names: [${head.join(", ")}]${featureNames.length > 5 ? "..." : ""}`);
  }

  const sample =
    featureNames.length > 0
      ? Object.fromEntries(featureNames.map((n) => [n, 0]))
      : Array(nFeatures || 1).fill(0);

  console.log(`\nPredict (single): input=${JSON.stringify(sample)}`);
  const r = await m.predict(sample);
  console.log(`  predictions: ${JSON.stringify(r.predictions)}`);
  console.log(`  probabilities: ${JSON.stringify(r.probabilities)}`);

  const label = await m.predictLabel(sample);
  console.log(`\npredictLabel: ${JSON.stringify(label)}`);

  if (taskType === "classification") {
    try {
      const proba = await m.predictProba(sample);
      console.log(`predictProba: ${JSON.stringify(proba)}`);
    } catch (e) {
      console.log(`predictProba unavailable: ${e.message}`);
    }
  }

  console.log(`\nPredict (batch of 3):`);
  const batch = [sample, sample, sample];
  const rb = await m.predictBatch(batch);
  console.log(`  predictions: ${JSON.stringify(rb.predictions)}`);

  console.log("\nOK");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
