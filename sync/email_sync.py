from __future__ import annotations

import email
import imaplib
from email.message import Message
from pathlib import Path
from typing import Optional

from logger import get_logger

log = get_logger(__name__)


class EmailSync:
    """Pull attachments from an IMAP mailbox into the local sandbox.
    Read-only — never sends, replies, or modifies server state."""

    def __init__(self, host: str, user: str, password: str, port: int = 993):
        self._host     = host
        self._user     = user
        self._password = password
        self._port     = port
        self._conn: Optional[imaplib.IMAP4_SSL] = None

    def _connect(self) -> imaplib.IMAP4_SSL:
        if self._conn is None:
            self._conn = imaplib.IMAP4_SSL(self._host, self._port)
            self._conn.login(self._user, self._password)
            log.info("IMAP connected | %s@%s", self._user, self._host)
        return self._conn

    def pull_attachments(
        self,
        local_dir: Path,
        folder: str = "INBOX",
        subject_filter: Optional[str] = None,
        max_emails: int = 50,
    ) -> list[Path]:
        """Download attachments from matching emails. Returns paths of saved files."""
        conn      = self._connect()
        criterion = '(SUBJECT "{}")'.format(subject_filter) if subject_filter else "ALL"

        conn.select(folder)
        _, ids  = conn.search(None, criterion)
        email_ids = ids[0].split()[-max_emails:]    # most recent N only

        saved = []
        for eid in email_ids:
            status, data = conn.fetch(eid, "(RFC822)")
            if status != "OK" or not data or not isinstance(data[0], tuple):
                log.warning("Skipping email id=%s: fetch returned status=%s", eid, status)
                continue
            msg = email.message_from_bytes(data[0][1])
            saved.extend(self._save_attachments(msg, local_dir))

        log.info("EmailSync | %d attachment(s) saved from %s", len(saved), folder)
        return saved

    def _save_attachments(self, msg: Message, dest: Path) -> list[Path]:
        saved = []
        for part in msg.walk():
            if part.get_content_disposition() != "attachment":
                continue
            filename = part.get_filename()
            if not filename:
                continue
            target = dest / filename
            target.write_bytes(part.get_payload(decode=True))
            log.debug("Saved attachment: %s", filename)
            saved.append(target)
        return saved

    def disconnect(self) -> None:
        if self._conn:
            try:
                self._conn.logout()
            except Exception:
                pass
            self._conn = None
            log.debug("IMAP disconnected")
