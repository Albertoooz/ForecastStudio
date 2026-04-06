"""
Azure Blob Storage wrapper — upload / download datasets, models, forecasts.

For local dev: falls back to file system if AZURE_STORAGE_CONNECTION_STRING is not set.
"""

import io
from pathlib import Path

import pandas as pd

from app.config import get_settings


class BlobStorage:
    """Abstraction over Azure Blob / local filesystem."""

    def __init__(self):
        self.settings = get_settings()
        self._client = None

        if self.settings.azure_storage_connection_string:
            try:
                from azure.storage.blob import BlobServiceClient

                self._client = BlobServiceClient.from_connection_string(
                    self.settings.azure_storage_connection_string
                )
            except Exception:
                pass  # Fall back to local FS

        # Local fallback directory
        self._local_root = Path(
            self.settings.local_storage_path
            if hasattr(self.settings, "local_storage_path")
            else ".storage"
        )

    @property
    def is_azure(self) -> bool:
        return self._client is not None

    # ── Upload ───────────────────────────────────────────────────────────

    def upload_bytes(self, container: str, blob_path: str, data: bytes) -> str:
        """Upload raw bytes → return full path."""
        if self.is_azure:
            assert self._client is not None
            blob_client = self._client.get_blob_client(container, blob_path)
            blob_client.upload_blob(data, overwrite=True)
            return f"{container}/{blob_path}"
        else:
            local_path = self._local_root / container / blob_path
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(data)
            return str(local_path)

    def upload_df(
        self, container: str, blob_path: str, df: pd.DataFrame, fmt: str = "parquet"
    ) -> str:
        """Upload DataFrame as parquet or CSV."""
        buf = io.BytesIO()
        if fmt == "parquet":
            df.to_parquet(buf, index=False)
        else:
            df.to_csv(buf, index=False)
        buf.seek(0)
        return self.upload_bytes(container, blob_path, buf.getvalue())

    def upload_model(self, blob_path: str, model_bytes: bytes) -> str:
        """Upload serialised model artifact."""
        return self.upload_bytes(self.settings.blob_container_models, blob_path, model_bytes)

    # ── Download ─────────────────────────────────────────────────────────

    def download_bytes(self, container: str, blob_path: str) -> bytes:
        """Download raw bytes."""
        if self.is_azure:
            assert self._client is not None
            blob_client = self._client.get_blob_client(container, blob_path)
            return blob_client.download_blob().readall()
        else:
            local_path = self._local_root / container / blob_path
            if not local_path.exists():
                raise FileNotFoundError(f"Blob not found: {local_path}")
            return local_path.read_bytes()

    def download_df(self, container: str, blob_path: str) -> pd.DataFrame:
        """Download blob → DataFrame."""
        data = self.download_bytes(container, blob_path)
        buf = io.BytesIO(data)
        if blob_path.endswith(".parquet"):
            return pd.read_parquet(buf)
        elif blob_path.endswith(".csv"):
            return pd.read_csv(buf)
        else:
            # Try parquet first, fallback to CSV
            try:
                return pd.read_parquet(buf)
            except Exception:
                buf.seek(0)
                return pd.read_csv(buf)

    def download_model(self, blob_path: str) -> bytes:
        """Download model artifact bytes."""
        return self.download_bytes(self.settings.blob_container_models, blob_path)

    # ── Delete ───────────────────────────────────────────────────────────

    def delete(self, container: str, blob_path: str) -> None:
        """Delete a blob."""
        if self.is_azure:
            assert self._client is not None
            blob_client = self._client.get_blob_client(container, blob_path)
            blob_client.delete_blob()
        else:
            local_path = self._local_root / container / blob_path
            if local_path.exists():
                local_path.unlink()

    # ── List ─────────────────────────────────────────────────────────────

    def list_blobs(self, container: str, prefix: str = "") -> list[str]:
        """List blob names under a prefix."""
        if self.is_azure:
            assert self._client is not None
            container_client = self._client.get_container_client(container)
            return [b.name for b in container_client.list_blobs(name_starts_with=prefix)]
        else:
            root = self._local_root / container
            if not root.exists():
                return []
            prefix_path = root / prefix if prefix else root
            return [str(p.relative_to(root)) for p in prefix_path.rglob("*") if p.is_file()]


# ── Singleton ────────────────────────────────────────────────────────────────

_blob_storage: BlobStorage | None = None


def get_blob_storage() -> BlobStorage:
    global _blob_storage
    if _blob_storage is None:
        _blob_storage = BlobStorage()
    return _blob_storage
