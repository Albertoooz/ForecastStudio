"""
Custom checkpointer serializer that handles non-JSON-serializable objects.

LangGraph's default MemorySaver uses msgpack, which rejects DataFrames,
fitted ML models (LightGBM, MLForecast), and Pydantic v2 models.

PickleSerde replaces msgpack with pickle so the graph can checkpoint
any Python object, including:
  - pd.DataFrame (training data, prepared data, future exog)
  - MLForecastModel (fitted model)
  - ForecastResult (Pydantic model)

Phase 1: pickle is fine — everything is in-process anyway.
Phase 3: swap for a proper serde that stores blobs in S3/GCS and only
         keeps references in PostgreSQL.
"""

from __future__ import annotations

import pickle
from typing import Any


class PickleSerde:
    """
    Minimal serde that uses pickle for all values.

    Satisfies langgraph's SerializerProtocol duck-type:
      dumps_typed(obj) -> (type_tag: str, raw_bytes: bytes)
      loads_typed((type_tag, raw_bytes)) -> obj
    """

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        return ("pickle", pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))

    def loads_typed(self, data: tuple[str, bytes]) -> Any:
        type_tag, raw = data
        if type_tag == "pickle":
            return pickle.loads(raw)
        # Safety net — should never be reached in Phase 1
        raise ValueError(f"PickleSerde: unknown type tag '{type_tag}'")
