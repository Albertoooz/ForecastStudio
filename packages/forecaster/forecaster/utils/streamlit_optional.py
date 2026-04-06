"""Stand-in for removed Streamlit UI: session_state-like object that returns defaults."""

from __future__ import annotations

from typing import Any


class _EmptySessionState:
    """Dict-like stand-in for legacy code that referenced st.session_state."""

    def get(self, key: str, default: Any = None) -> Any:
        return default

    def __contains__(self, key: str) -> bool:
        return False

    def __getitem__(self, key: str) -> Any:
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        return

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(name)


_SESSION = _EmptySessionState()


def get_session_state() -> _EmptySessionState:
    return _SESSION
