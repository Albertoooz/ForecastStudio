def test_workspace_imports() -> None:
    """Sanity check — backend package and forecaster import."""
    import app.main  # noqa: F401

    import forecaster  # noqa: F401
