from __future__ import annotations

from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

from logger import get_logger

log = get_logger(__name__)


# Interface every cloud connector must satisfy
@runtime_checkable
class CloudConnector(Protocol):
    def list_remote(self, prefix: str = "") -> list[str]: ...
    def pull(self, remote_path: str, local_dir: Path) -> Path: ...


class S3Connector:
    """Pull-only S3 connector. Requires boto3 and AWS credentials in environment."""

    def __init__(self, bucket: str, prefix: str = ""):
        self.bucket = bucket
        self.prefix = prefix
        self._s3    = None      # lazy boto3 client

    def _client(self):
        if self._s3 is None:
            import boto3
            self._s3 = boto3.client("s3")
            log.debug("S3 client initialised | bucket=%s", self.bucket)
        return self._s3

    def list_remote(self, prefix: str = "") -> list[str]:
        resp = self._client().list_objects_v2(Bucket=self.bucket, Prefix=self.prefix + prefix)
        return [obj["Key"] for obj in resp.get("Contents", [])]

    def pull(self, remote_path: str, local_dir: Path) -> Path:
        local_path = local_dir / Path(remote_path).name
        log.info("S3 pull | %s → %s", remote_path, local_path)
        self._client().download_file(self.bucket, remote_path, str(local_path))
        return local_path


class GDriveConnector:
    """Pull-only Google Drive connector. Requires google-api-python-client."""

    def __init__(self, credentials_path: str):
        self._creds_path = credentials_path
        self._svc        = None     # lazy Drive service

    def _service(self):
        if self._svc is None:
            from googleapiclient.discovery import build
            from google.oauth2.credentials import Credentials
            self._svc = build("drive", "v3", credentials=Credentials.from_authorized_user_file(self._creds_path))
            log.debug("Google Drive service initialised")
        return self._svc

    def list_remote(self, prefix: str = "") -> list[str]:
        q       = "name contains '{}'".format(prefix) if prefix else None
        results = self._service().files().list(q=q, fields="files(id,name)").execute()
        return [f["id"] for f in results.get("files", [])]

    def pull(self, remote_path: str, local_dir: Path) -> Path:
        from googleapiclient.http import MediaIoBaseDownload
        # remote_path is a Drive file ID
        meta       = self._service().files().get(fileId=remote_path, fields="name").execute()
        local_path = local_dir / meta["name"]
        with open(local_path, "wb") as fh:
            dl   = MediaIoBaseDownload(fh, self._service().files().get_media(fileId=remote_path))
            done = False
            while not done:
                _, done = dl.next_chunk()
        log.info("GDrive pull | %s → %s", remote_path, local_path)
        return local_path


class CloudSync:
    """Orchestrates pull-only sync from registered connectors into the local sandbox."""

    def __init__(self, local_dir: Path):
        self.local_dir   = local_dir
        self._connectors: dict[str, CloudConnector] = {}

    def register(self, name: str, connector: CloudConnector) -> None:
        self._connectors[name] = connector
        log.info("Connector registered: %s", name)

    def pull_all(self, source: Optional[str] = None) -> list[Path]:
        """Pull from all connectors, or only the named one. Returns saved file paths."""
        if source and source not in self._connectors:
            raise ValueError(
                "Unknown connector '{}'. Register it first with register_s3() or register_gdrive().".format(source)
            )
        targets = {source: self._connectors[source]} if source else self._connectors
        pulled  = []

        for name, conn in targets.items():
            log.info("Syncing from: %s", name)
            for remote in conn.list_remote():
                try:
                    pulled.append(conn.pull(remote, self.local_dir))
                except Exception as exc:
                    log.error("Pull failed | %s | %s: %s", name, remote, exc, exc_info=True)

        log.info("Sync complete | %d file(s) pulled", len(pulled))
        return pulled
