"""MathExec API client."""
from typing import Any, Dict, List, Optional, Union

import requests

DEFAULT_BASE_URL = "https://api.mathexec.com"


class MathExecError(Exception):
    """Error from the MathExec API."""


class Model:
    """A published MathExec model.

    Construct instances via :meth:`Model.load`, which mirrors the pattern used
    by other ML SDKs (HuggingFace ``from_pretrained``, Keras ``load_model``,
    joblib ``load``). The direct ``Model(...)`` constructor is internal; it
    does not fetch or validate the model — use ``Model.load()`` for that.
    """

    def __init__(
        self,
        model_id: str,
        base_url: str = DEFAULT_BASE_URL,
        api_key: Optional[str] = None,
    ):
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._info: Optional[dict] = None

    @classmethod
    def load(
        cls,
        model_id: str,
        base_url: str = DEFAULT_BASE_URL,
        api_key: Optional[str] = None,
    ) -> "Model":
        """Fetch a published model by ID.

        This is the canonical way to obtain a Model. It eagerly resolves the
        model's metadata so a bad ID fails fast at load time, not at first
        predict() call.

        Args:
            model_id: Published model identifier. Accepts the short form
                (``"a3k9x2qm"``) or the fully-qualified ``"project/experiment"``
                form (``"churn-demo/experiment-3"``) — the server resolves the
                canonical ID.
            base_url: API base URL (override for self-hosted deployments).
            api_key: Bearer token for private / rate-limited models.

        Returns:
            A :class:`Model` ready for prediction.

        Raises:
            MathExecError: If the model does not exist or is not accessible.

        Example:
            >>> from mathexec import Model
            >>> model = Model.load("project-id/experiment-3")
            >>> model.predict({"logins": 4, "tickets": 7, "features_used": 2, "tenure": 3})
        """
        instance = cls(model_id, base_url=base_url, api_key=api_key)
        # Touch .info to validate the model exists. This is the "fail fast"
        # step that distinguishes load() from bare construction.
        _ = instance.info
        return instance

    @property
    def info(self) -> dict:
        """Lazy-loaded model metadata."""
        if self._info is None:
            self._info = self._get(f"/models/{self.model_id}")
        return self._info

    def predict(
        self, input: Union[Dict[str, Any], List[float]]
    ) -> dict:
        """Predict on a single sample.

        Returns the full response dict so callers can inspect probabilities
        and target_metadata. For just the label, see :meth:`predict_label`.
        For just the probability, see :meth:`predict_proba`.

        Args:
            input: Either a dict of named features ``{"age": 25, "income": 50000}``
                or a list of positional feature values ``[25, 50000]``.

        Returns:
            Dict with keys ``predictions``, ``probabilities``, ``count``,
            ``model_id``, ``model_name``, ``target_metadata``.

            For models trained on string labels (``"yes"``/``"no"``,
            multiclass names), ``predictions`` already contains the
            original labels — no decoding needed on the client.
        """
        body: dict = {}
        if isinstance(input, dict):
            body["input"] = input
        else:
            body["features"] = input
        return self._post(f"/m/{self.model_id}/predict", body)

    def predict_batch(
        self, inputs: Union[List[Dict[str, Any]], List[List[float]]]
    ) -> dict:
        """Predict on multiple samples.

        Args:
            inputs: List of dicts (named) or list of lists (positional).

        Returns:
            Dict with ``predictions``, ``probabilities``, ``count``,
            ``target_metadata``. Predictions are in the user's original
            label space (string labels survive the round trip).
        """
        body: dict = {}
        if inputs and isinstance(inputs[0], dict):
            body["inputs"] = inputs
        else:
            body["batch"] = inputs
        return self._post(f"/m/{self.model_id}/predict", body)

    # ------------------------------------------------------------------
    # sklearn-style accessors: predict_label / predict_proba
    # ------------------------------------------------------------------

    def predict_label(
        self, input: Union[Dict[str, Any], List[float]]
    ) -> Any:
        """Return just the predicted label for a single sample.

        For classification on string labels, this returns the original
        label (``"yes"``, ``"churn"``, ``"cat"``, ...). For numeric
        classes, an ``int``. For regression, a ``float``.

        >>> model = Model.load("project/exp")
        >>> model.predict_label({"age": 32, "balance": 1200})
        'yes'
        """
        resp = self.predict(input)
        preds = resp.get("predictions") or []
        return preds[0] if preds else None

    def predict_proba(
        self,
        input: Union[Dict[str, Any], List[float], List[Dict[str, Any]], List[List[float]]],
        *,
        positive_class: Any = None,
    ) -> Union[float, List[float], None]:
        """Return prediction probabilities (classification only).

        For binary classification the API returns a single probability per
        row — model confidence in the positive class (the label encoded
        as ``1`` during training; usually the lexicographically larger of
        the two for string targets, e.g. ``"yes"`` over ``"no"``).

        For multiclass models the API currently returns the top-1
        probability per row; pass ``positive_class`` to filter the rows
        whose prediction matches that class (the others get ``None``) so
        callers can build per-class scores without recomputing.

        Args:
            input: Single sample (dict or list) or batch (list of dicts /
                list of lists). Auto-detected from shape.
            positive_class: Optional class to filter for in multiclass
                outputs. Ignored for binary models.

        Returns:
            ``float`` for a single sample, ``list[float|None]`` for a
            batch, or ``None`` if the model isn't a classifier (caller
            can branch on this without checking ``target_metadata``).

        Raises:
            MathExecError: If the model returned no probabilities at all
                (i.e. it's a regression model).
        """
        is_batch = (
            isinstance(input, list)
            and len(input) > 0
            and isinstance(input[0], (list, dict))
        )
        resp = self.predict_batch(input) if is_batch else self.predict(input)  # type: ignore[arg-type]

        probs = resp.get("probabilities")
        meta = resp.get("target_metadata") or {}
        if probs is None:
            if meta.get("task_type") == "regression":
                raise MathExecError(
                    "predict_proba is undefined for regression models. "
                    "Use predict() to get the numeric output."
                )
            raise MathExecError(
                "Model returned no probabilities. The trained model may "
                "not expose predict_proba (e.g. some custom formulas)."
            )

        # Multiclass: optionally mask rows whose top-1 doesn't match
        # the requested class. Binary models ignore this argument.
        if positive_class is not None and not meta.get("is_binary", False):
            preds = resp.get("predictions") or []
            masked = [p if pred == positive_class else None for pred, p in zip(preds, probs)]
            return masked[0] if not is_batch else masked

        return probs[0] if not is_batch else probs

    @property
    def task_type(self) -> Optional[str]:
        """``"classification"`` or ``"regression"``. ``None`` for legacy models."""
        return (self.info.get("target_metadata") or {}).get("task_type") or self.info.get("task_type")

    @property
    def class_labels(self) -> List[Any]:
        """Original target labels in encoded order (empty for regression)."""
        return list((self.info.get("target_metadata") or {}).get("class_labels") or [])

    @property
    def positive_class(self) -> Any:
        """For binary classification, the label encoded as ``1`` (e.g. ``"yes"``)."""
        meta = self.info.get("target_metadata") or {}
        labels = meta.get("class_labels") or []
        if meta.get("is_binary") and len(labels) >= 2:
            return labels[1]
        return None

    def _headers(self) -> dict:
        h: Dict[str, str] = {}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _get(self, path: str) -> dict:
        resp = requests.get(f"{self.base_url}{path}", headers=self._headers())
        if resp.status_code == 429:
            raise MathExecError("Rate limited. Wait and retry.")
        if resp.status_code == 404:
            raise MathExecError(f"Model '{self.model_id}' not found.")
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, body: dict) -> dict:
        headers = {**self._headers(), "Content-Type": "application/json"}
        resp = requests.post(
            f"{self.base_url}{path}", json=body, headers=headers
        )
        if resp.status_code == 429:
            raise MathExecError("Rate limited. Wait and retry.")
        if resp.status_code == 404:
            raise MathExecError(f"Model '{self.model_id}' not found.")
        if resp.status_code == 400:
            detail = resp.json().get("detail", resp.text)
            raise MathExecError(f"Bad request: {detail}")
        resp.raise_for_status()
        return resp.json()

    def __repr__(self) -> str:
        return f"mathexec.Model('{self.model_id}')"


def model(
    model_id: str,
    base_url: str = DEFAULT_BASE_URL,
    api_key: Optional[str] = None,
) -> Model:
    """Deprecated. Use :meth:`Model.load` instead.

    Kept as a thin alias during 0.1.x so existing imports keep working, but
    all documentation and examples now promote ``Model.load(...)``.
    """
    import warnings
    warnings.warn(
        "mathexec.model() is deprecated; use `from mathexec import Model; "
        "Model.load(model_id)` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return Model.load(model_id, base_url=base_url, api_key=api_key)


def list_models(
    base_url: str = DEFAULT_BASE_URL,
    api_key: Optional[str] = None,
) -> list:
    """List your published models (requires API key).

    Returns:
        List of model summary dicts.
    """
    headers: Dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    resp = requests.get(f"{base_url}/models/my", headers=headers)
    resp.raise_for_status()
    return resp.json().get("models", [])
