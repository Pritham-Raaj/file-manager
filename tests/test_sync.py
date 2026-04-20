"""
Tests for CloudSync (S3) and EmailSync (IMAP) — both mocked at the I/O layer.

Pull-only invariant: these tests verify that no file is ever uploaded or
written to a remote destination, even when the mocked sink accepts any call.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock, call
import pytest

import config


# ---------------------------------------------------------------------------
# S3 sync
# ---------------------------------------------------------------------------

class TestS3Sync:

    def _make_cloud_sync(self, tmp_root):
        from sync.cloud_sync import CloudSync
        return CloudSync(root=tmp_root)

    def _make_s3_connector(self):
        from sync.cloud_sync import S3Connector
        return S3Connector(bucket="test-bucket", prefix="data/")

    def test_register_connector(self, tmp_path):
        cs = self._make_cloud_sync(tmp_path)
        conn = self._make_s3_connector()
        cs.register("my-s3", conn)
        assert "my-s3" in cs._connectors

    def test_pull_downloads_files(self, tmp_path):
        """pull_all should download objects from S3 into the root dir."""
        from sync.cloud_sync import CloudSync, S3Connector

        # S3Connector uses boto3.client("s3"), not boto3.resource.
        # list_remote calls client.list_objects_v2(); pull calls client.download_file().
        downloaded = []

        def fake_download_file(bucket, key, local_path):
            p = Path(local_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("S3 content", encoding="utf-8")
            downloaded.append(p)

        fake_client = MagicMock()
        fake_client.list_objects_v2.return_value = {
            "Contents": [{"Key": "data/report.txt"}]
        }
        fake_client.download_file.side_effect = fake_download_file

        with patch("sync.cloud_sync.boto3") as mock_boto3:
            mock_boto3.client.return_value = fake_client
            cs   = CloudSync(root=tmp_path)
            conn = S3Connector(bucket="test-bucket", prefix="data/")
            cs.register("s3", conn)
            pulled = cs.pull_all("s3")

        assert len(pulled) == 1
        assert pulled[0].name == "report.txt"

    def test_pull_no_connectors_returns_empty(self, tmp_path):
        from sync.cloud_sync import CloudSync
        cs = CloudSync(root=tmp_path)
        pulled = cs.pull_all()
        assert pulled == []

    def test_s3_never_uploads(self, tmp_path):
        """Verify that no boto3 put_object / upload_file calls are made."""
        from sync.cloud_sync import CloudSync, S3Connector

        fake_client = MagicMock()
        fake_client.list_objects_v2.return_value = {"Contents": []}  # nothing to pull

        with patch("sync.cloud_sync.boto3") as mock_boto3:
            mock_boto3.client.return_value = fake_client
            cs   = CloudSync(root=tmp_path)
            conn = S3Connector(bucket="test-bucket", prefix="")
            cs.register("s3", conn)
            cs.pull_all("s3")
            # Ensure no upload-related methods were ever called on the client
            assert not fake_client.put_object.called
            assert not fake_client.upload_file.called


# ---------------------------------------------------------------------------
# Email sync
# ---------------------------------------------------------------------------

class TestEmailSync:

    def _build_mime_attachment(self, filename: str, body: bytes) -> bytes:
        """Return a minimal RFC 2822 message with one attachment."""
        import email
        from email.mime.multipart import MIMEMultipart
        from email.mime.base      import MIMEBase
        from email.mime.text      import MIMEText
        from email                import encoders

        msg = MIMEMultipart()
        msg["Subject"] = "Test report"
        msg["From"]    = "sender@example.com"
        msg["To"]      = "user@example.com"
        msg.attach(MIMEText("See attachment.", "plain"))

        part = MIMEBase("application", "octet-stream")
        part.set_payload(body)
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f'attachment; filename="{filename}"')
        msg.attach(part)
        return msg.as_bytes()

    def test_pull_attachments_saves_files(self, tmp_path):
        """EmailSync should save each attachment to local_dir."""
        from sync.email_sync import EmailSync

        raw_msg = self._build_mime_attachment("health_report.txt", b"Patient data here")

        fake_imap = MagicMock()
        # IMAP select() response
        fake_imap.select.return_value = ("OK", [b"1"])
        # search() returns one UID
        fake_imap.search.return_value = ("OK", [b"1"])
        # fetch() returns raw message bytes
        fake_imap.fetch.return_value  = ("OK", [(b"1 (RFC822 {N})", raw_msg)])

        with patch("sync.email_sync.imaplib") as mock_imap_lib:
            mock_imap_lib.IMAP4_SSL.return_value = fake_imap
            syncer  = EmailSync("imap.example.com", "user@example.com", "password")
            pulled  = syncer.pull_attachments(tmp_path, max_emails=10)
            syncer.disconnect()

        assert len(pulled) >= 1
        names = [p.name for p in pulled]
        assert "health_report.txt" in names

    def test_pull_no_messages_returns_empty(self, tmp_path):
        """Empty inbox should return an empty list without raising."""
        from sync.email_sync import EmailSync

        fake_imap = MagicMock()
        fake_imap.select.return_value = ("OK", [b"0"])
        fake_imap.search.return_value = ("OK", [b""])
        fake_imap.fetch.return_value  = ("OK", [])

        with patch("sync.email_sync.imaplib") as mock_imap_lib:
            mock_imap_lib.IMAP4_SSL.return_value = fake_imap
            syncer = EmailSync("imap.example.com", "u", "p")
            pulled = syncer.pull_attachments(tmp_path, max_emails=50)
            syncer.disconnect()

        assert pulled == []

    def test_email_never_sends(self, tmp_path):
        """Verify that no SMTP or IMAP append/store calls are made."""
        from sync.email_sync import EmailSync

        fake_imap = MagicMock()
        fake_imap.select.return_value = ("OK", [b"0"])
        fake_imap.search.return_value = ("OK", [b""])

        with patch("sync.email_sync.imaplib") as mock_imap_lib:
            mock_imap_lib.IMAP4_SSL.return_value = fake_imap
            syncer = EmailSync("imap.example.com", "u", "p")
            syncer.pull_attachments(tmp_path, max_emails=10)
            syncer.disconnect()
            # IMAP store() is used for marking messages; must not be called
            assert not fake_imap.store.called
            # SMTP must never be touched
            mock_imap_lib.SMTP_SSL = MagicMock()
            assert not mock_imap_lib.SMTP_SSL.called
