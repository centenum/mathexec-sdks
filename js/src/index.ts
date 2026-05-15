/**
 * MathExec JavaScript SDK — call published model endpoints.
 */

const DEFAULT_BASE_URL = "https://api.mathexec.com";

export interface PredictResult {
  predictions: any[];
  probabilities?: number[];
  count: number;
  model_id: string;
  model_name: string;
  target_metadata?: {
    task_type?: string;
    is_binary?: boolean;
    class_labels?: any[];
  };
  _meta?: { powered_by: string; model_url: string };
}

export interface ModelInfo {
  model_id: string;
  name: string;
  description: string;
  formula_latex: string;
  model_type: string;
  task_type: string;
  author_name: string;
  feature_names: string[] | null;
  n_features: number;
  metrics: Record<string, any>;
  predict_count: number;
  view_count: number;
  created_at: string;
  target_metadata?: {
    task_type?: string;
    is_binary?: boolean;
    class_labels?: any[];
  };
}

export interface ModelOptions {
  baseUrl?: string;
  apiKey?: string;
}

export class MathExecError extends Error {
  constructor(
    message: string,
    public statusCode?: number
  ) {
    super(message);
    this.name = "MathExecError";
  }
}

export class Model {
  private _info: ModelInfo | null = null;

  constructor(
    public readonly modelId: string,
    private baseUrl: string = DEFAULT_BASE_URL,
    private apiKey?: string
  ) {
    this.baseUrl = baseUrl.replace(/\/$/, "");
  }

  /**
   * Fetch a published model by ID. Eagerly resolves metadata so a bad ID
   * fails fast at load() time, not at first predict() call.
   *
   * @example
   * ```ts
   * import { Model } from 'mathexec-sdk'
   * const m = await Model.load('a3k9x2qm')
   * await m.predict({ age: 25, income: 50000 })
   * ```
   */
  static async load(modelId: string, opts?: ModelOptions): Promise<Model> {
    const m = new Model(modelId, opts?.baseUrl, opts?.apiKey);
    await m.info();
    return m;
  }

  /** Get model metadata (lazy-loaded, cached). */
  async info(): Promise<ModelInfo> {
    if (!this._info) {
      this._info = await this.get<ModelInfo>(`/models/${this.modelId}`);
    }
    return this._info;
  }

  /**
   * Predict on a single sample.
   * @param input Named features `{age: 25}` or positional array `[25, 50000]`.
   */
  async predict(
    input: Record<string, any> | number[]
  ): Promise<PredictResult> {
    const body = Array.isArray(input) ? { features: input } : { input };
    return this.post<PredictResult>(`/m/${this.modelId}/predict`, body);
  }

  /**
   * Predict on multiple samples.
   * @param inputs Array of named dicts or array of number arrays.
   */
  async predictBatch(
    inputs: Record<string, any>[] | number[][]
  ): Promise<PredictResult> {
    const isNamed =
      inputs.length > 0 &&
      typeof inputs[0] === "object" &&
      !Array.isArray(inputs[0]);
    const body = isNamed ? { inputs } : { batch: inputs };
    return this.post<PredictResult>(`/m/${this.modelId}/predict`, body);
  }

  /**
   * Return just the predicted label for a single sample.
   *
   * For classification on string labels this returns the original label
   * (`'yes'`, `'churn'`, ...). Numeric classes return a `number`.
   * Regression returns a `number`.
   */
  async predictLabel(
    input: Record<string, any> | number[]
  ): Promise<any> {
    const r = await this.predict(input);
    return r.predictions && r.predictions.length > 0 ? r.predictions[0] : null;
  }

  /**
   * Return prediction probabilities (classification only).
   *
   * For binary classification the API returns one probability per row —
   * model confidence in the positive class (the label encoded as `1`,
   * usually the lexicographically larger of the two for string targets).
   *
   * For multiclass models the API returns the top-1 probability per row.
   * Pass `positiveClass` to mask rows whose prediction doesn't match
   * (those become `null`), so you can build per-class scores cheaply.
   *
   * @throws MathExecError if the model is regression or returns no probs.
   */
  async predictProba(
    input:
      | Record<string, any>
      | number[]
      | Record<string, any>[]
      | number[][],
    opts?: { positiveClass?: any }
  ): Promise<number | (number | null)[] | null> {
    const isBatch =
      Array.isArray(input) &&
      input.length > 0 &&
      (Array.isArray(input[0]) || typeof input[0] === "object");

    const r = isBatch
      ? await this.predictBatch(input as any)
      : await this.predict(input as any);

    const probs = r.probabilities;
    const meta = r.target_metadata || {};

    if (probs == null) {
      if (meta.task_type === "regression") {
        throw new MathExecError(
          "predictProba is undefined for regression models. Use predict() to get the numeric output."
        );
      }
      throw new MathExecError(
        "Model returned no probabilities. The trained model may not expose predict_proba."
      );
    }

    if (opts?.positiveClass !== undefined && !meta.is_binary) {
      const preds = r.predictions || [];
      const masked = preds.map((p: any, i: number) =>
        p === opts.positiveClass ? probs[i] : null
      );
      return isBatch ? masked : masked[0];
    }

    return isBatch ? probs : probs[0];
  }

  /** `'classification'` or `'regression'` (or null for legacy models). */
  async taskType(): Promise<string | null> {
    const info = await this.info();
    return info.target_metadata?.task_type || info.task_type || null;
  }

  /** Original target labels in encoded order (empty for regression). */
  async classLabels(): Promise<any[]> {
    const info = await this.info();
    return info.target_metadata?.class_labels || [];
  }

  /** For binary classification, the label encoded as `1` (e.g. `'yes'`). */
  async positiveClass(): Promise<any> {
    const info = await this.info();
    const meta = info.target_metadata;
    const labels = meta?.class_labels || [];
    if (meta?.is_binary && labels.length >= 2) return labels[1];
    return null;
  }

  private async get<T>(path: string): Promise<T> {
    const headers: Record<string, string> = {};
    if (this.apiKey) headers["Authorization"] = `Bearer ${this.apiKey}`;

    const res = await fetch(`${this.baseUrl}${path}`, { headers });
    if (res.status === 429) throw new MathExecError("Rate limited. Wait and retry.", 429);
    if (res.status === 404) throw new MathExecError(`Model '${this.modelId}' not found.`, 404);
    if (!res.ok) throw new MathExecError(`API error: ${res.status}`, res.status);
    return res.json();
  }

  private async post<T>(path: string, body: any): Promise<T> {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };
    if (this.apiKey) headers["Authorization"] = `Bearer ${this.apiKey}`;

    const res = await fetch(`${this.baseUrl}${path}`, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
    });
    if (res.status === 429) throw new MathExecError("Rate limited. Wait and retry.", 429);
    if (res.status === 404) throw new MathExecError(`Model '${this.modelId}' not found.`, 404);
    if (res.status === 400) {
      const data = await res.json().catch(() => ({}));
      throw new MathExecError(`Bad request: ${data.detail || res.statusText}`, 400);
    }
    if (!res.ok) throw new MathExecError(`API error: ${res.status}`, res.status);
    return res.json();
  }
}

/**
 * Get a published model by ID (no fail-fast check). Prefer `Model.load()`
 * for the canonical pattern; this factory is kept for backwards compat.
 */
export function model(
  modelId: string,
  opts?: ModelOptions
): Model {
  return new Model(modelId, opts?.baseUrl, opts?.apiKey);
}

export interface ModelSummary {
  model_id: string;
  name: string;
  task_type?: string;
  created_at?: string;
  [key: string]: any;
}

/**
 * List the caller's published models. Requires an API key.
 *
 * @example
 * ```ts
 * import { listModels } from 'mathexec-sdk'
 * const models = await listModels({ apiKey: 'mx_live_...' })
 * ```
 */
export async function listModels(
  opts?: ModelOptions
): Promise<ModelSummary[]> {
  const baseUrl = (opts?.baseUrl || DEFAULT_BASE_URL).replace(/\/$/, "");
  const headers: Record<string, string> = {};
  if (opts?.apiKey) headers["Authorization"] = `Bearer ${opts.apiKey}`;

  const res = await fetch(`${baseUrl}/models/my`, { headers });
  if (res.status === 401)
    throw new MathExecError("Unauthorized. Pass apiKey.", 401);
  if (!res.ok)
    throw new MathExecError(`API error: ${res.status}`, res.status);
  const data = await res.json();
  return data.models || [];
}
