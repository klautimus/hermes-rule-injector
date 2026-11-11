"""Semantic matcher using sentence-transformers/all-MiniLM-L6-v2.

**INERT IN THIS INSTALL (verified 2026-07-27).** The active venv at
`/home/kevin/.hermes/hermes-agent/venv/bin/python3` does not have
`sentence_transformers` or `numpy` installed. `_ensure_model` catches
`ImportError` and sets `_import_failed=True` permanently for the
process. Semantic rule branches silently never fire, with no error and
no log line.

The review-only LLM adjudicator (`auxiliary.adjudicator`, provider
`minimax`, model `minimax-m3`) is the live context-preservation
mechanism in this install. See
`~/.hermes/plugins/rule-injector/docs/adjudicator-replaces-semantic.md`
for the production architecture.

Design notes (preserved from the original design doc, for the case
where sentence-transformers is later installed):

  - Lazy singleton: model loads only on first semantic rule evaluation.
  - Deferred import: torch/sentence-transformers imported inside _ensure_model()
    so the import cost (~3-5s) is paid only if semantic rules exist AND fire.
  - LRU query cache: user message embeddings are cached (temporal locality —
    same message hits pre_llm_call, then pre_verify).
  - Reference embeddings: cached at first match in CompiledRule, not at compile
    time, to preserve true lazy loading.
  - Graceful degradation: if sentence-transformers isn't installed or model
    fails to load, semantic rules silently never fire. Keyword rules unaffected.

  The CUDA_VISIBLE_DEVICES="" env var must be set BEFORE torch initializes,
  so it is blanked inside _ensure_model() around the deferred import (torch has
  not been imported at module load time). It is then RESTORED: os.environ is
  inherited by every child process the gateway spawns, so leaving it set would
  disable CUDA for the whole process tree.
"""

import os
import hashlib
import logging
import threading
from collections import OrderedDict
from typing import Optional, List

logger = logging.getLogger(__name__)


class SemanticMatcher:
    """Thread-safe singleton for lazy sentence-transformer model loading.

    The model loads only when a semantic rule is first evaluated. If
    sentence-transformers isn't installed, the instance marks itself
    permanently unavailable and all semantic rules silently skip.
    """

    _instance: Optional["SemanticMatcher"] = None
    _lock = threading.Lock()

    def __init__(self):
        self._model = None
        self._import_failed = False
        self._query_cache: OrderedDict = OrderedDict()  # md5(text) -> np.ndarray
        self._max_cache_size = 256

    @classmethod
    def get_instance(cls) -> "SemanticMatcher":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _ensure_model(self):
        """Lazy import + model load. Returns None if unavailable.

        Temporarily blank CUDA_VISIBLE_DEVICES across the torch import to keep
        this embedding model on CPU, then restore it. The variable must NOT be
        left set: os.environ is inherited by every child process the gateway
        spawns, and a lingering CUDA_VISIBLE_DEVICES="" makes cuInit() return
        CUDA_ERROR_NO_DEVICE for all of them (torch, ffmpeg, ollama, builds).
        The GPU on this host is functional -- NVENC encodes and NVDEC decodes
        fine -- so forcing CPU here is a per-model choice, not a hardware limit.

        Environment bypass: Set RULE_INJECTOR_NO_SEMANTIC=1 to skip semantic
        matching entirely. Used by test suite to avoid scipy/sklearn import hang
        (sentence_transformers → sklearn → scipy → _make_sphinx_capabilities hangs).
        """
        if os.environ.get("RULE_INJECTOR_NO_SEMANTIC"):
            self._import_failed = True
            return None
        if self._import_failed:
            return None
        if self._model is not None:
            return self._model

        with self._lock:
            if self._model is not None:
                return self._model
            if self._import_failed:
                return None
            try:
                # Force CPU mode for this one model, then restore.
                #
                # CUDA_VISIBLE_DEVICES must be empty BEFORE torch is imported,
                # because torch caches CUDA availability at import time. But
                # os.environ is INHERITED BY EVERY CHILD PROCESS Hermes spawns
                # (terminal commands, subagents, builds) -- leaving it set to ""
                # globally silently disables CUDA for the entire gateway tree and
                # makes cuInit() return CUDA_ERROR_NO_DEVICE everywhere.
                # So: set it only across the import, then put it back.
                _prev_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
                try:
                    from sentence_transformers import SentenceTransformer
                finally:
                    if _prev_cvd is None:
                        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                    else:
                        os.environ["CUDA_VISIBLE_DEVICES"] = _prev_cvd

                cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
                self._model = SentenceTransformer(
                    "sentence-transformers/all-MiniLM-L6-v2",
                    cache_folder=cache_dir,
                    device="cpu",
                )
                logger.info(
                    "rule-injector: semantic model loaded (all-MiniLM-L6-v2, CPU)"
                )
            except Exception as e:
                self._import_failed = True
                logger.warning(
                    "rule-injector: sentence-transformers not available, "
                    "semantic rules will be inactive (converted to keyword "
                    "triggers 2026-07-31; this fires only if a semantic rule "
                    "is re-added): %s",
                    e,
                )
                return None
        return self._model

    def encode_query(self, text: str):
        """Encode user text with LRU cache. Returns normalized np.ndarray or None.

        Uses normalize_embeddings=True so cosine similarity = dot product,
        eliminating the need to compute norms at match time.
        """
        if self._import_failed:
            return None

        key = hashlib.md5(text.encode("utf-8")).hexdigest()
        if key in self._query_cache:
            self._query_cache.move_to_end(key)
            return self._query_cache[key]

        model = self._ensure_model()
        if model is None:
            return None

        try:
            embedding = model.encode(
                text,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception:
            return None

        # Store in LRU cache
        self._query_cache[key] = embedding
        self._query_cache.move_to_end(key)

        if len(self._query_cache) > self._max_cache_size:
            self._query_cache.popitem(last=False)

        return embedding

    def encode_references(self, references: List[str]):
        """Batch-encode reference phrases. Returns normalized np.ndarray or None.

        Called lazily on first match of a semantic rule to back-fill
        reference embeddings into the CompiledRule.
        """
        model = self._ensure_model()
        if model is None:
            return None

        try:
            return model.encode(
                references,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception:
            return None

    def is_available(self) -> bool:
        """Check if the model is available without forcing a load attempt."""
        return not self._import_failed and (
            self._model is not None or self._ensure_model() is not None
        )
