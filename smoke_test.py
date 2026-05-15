"""Smoke test: hit a published model with the Python SDK.

Usage:
    cd sdk/python && pip install -e .
    cd ../..
    MODEL_ID=abc12345 BASE_URL=http://localhost:8001 python sdk/smoke_test.py

Set API_KEY=mx_live_... if the model is private/unlisted.
"""
import os
import sys

from mathexec import Model, MathExecError


def main() -> int:
    model_id = os.environ.get("MODEL_ID")
    base_url = os.environ.get("BASE_URL", "http://localhost:8001")
    api_key = os.environ.get("API_KEY")

    if not model_id:
        print("Set MODEL_ID=<your published model id>", file=sys.stderr)
        return 2

    print(f"Loading {model_id} from {base_url}...")
    try:
        m = Model.load(model_id, base_url=base_url, api_key=api_key)
    except MathExecError as e:
        print(f"Load failed: {e}", file=sys.stderr)
        return 1

    info = m.info
    feature_names = info.get("feature_names") or []
    n_features = info.get("n_features") or len(feature_names)
    task_type = m.task_type

    print(f"  name: {info.get('name')}")
    print(f"  task: {task_type}, features: {n_features}")
    if feature_names:
        print(f"  feature_names: {feature_names[:5]}{'...' if len(feature_names) > 5 else ''}")

    sample = (
        {n: 0 for n in feature_names}
        if feature_names
        else [0] * (n_features or 1)
    )

    print(f"\nPredict (single): input={sample}")
    r = m.predict(sample)
    print(f"  predictions: {r.get('predictions')}")
    print(f"  probabilities: {r.get('probabilities')}")

    label = m.predict_label(sample)
    print(f"\npredict_label: {label}")

    if task_type == "classification":
        try:
            proba = m.predict_proba(sample)
            print(f"predict_proba: {proba}")
        except MathExecError as e:
            print(f"predict_proba unavailable: {e}")

    print(f"\nPredict (batch of 3):")
    batch = [sample, sample, sample]
    r = m.predict_batch(batch)
    print(f"  predictions: {r.get('predictions')}")

    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
