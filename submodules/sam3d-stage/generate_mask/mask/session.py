"""In-memory LRU session store for image embeddings.

Each session holds: state (dict from MaskPipeline.embed()),
image (PIL for re-prediction), image_bytes (for sha hashing), last_seen (timestamp).

Max sessions: 16 (configurable). TTL: 30 min idle. Evicts oldest on overflow.
"""
from __future__ import annotations
import time
import uuid
from dataclasses import dataclass

from PIL import Image


@dataclass
class SessionEntry:
    session_id: str
    state: dict
    image: Image.Image
    image_bytes: bytes
    width: int
    height: int
    created_at: float
    last_seen: float


class SessionManager:
    def __init__(self,
                 max_sessions: int = 16,
                 ttl_seconds: int = 1800):
        self._sessions: dict[str, SessionEntry] = {}
        self._max = max_sessions
        self._ttl = ttl_seconds

    def create(self,
               image: Image.Image,
               image_bytes: bytes,
               state: dict) -> str:
        """Create a new session and return its id."""
        self.evict_stale()
        while len(self._sessions) >= self._max:
            oldest_id = min(
                self._sessions, key=lambda k: self._sessions[k].last_seen
            )
            del self._sessions[oldest_id]

        sid = uuid.uuid4().hex
        now = time.time()
        self._sessions[sid] = SessionEntry(
            session_id=sid,
            state=state,
            image=image,
            image_bytes=image_bytes,
            width=image.width,
            height=image.height,
            created_at=now,
            last_seen=now,
        )
        return sid

    def get(self, session_id: str) -> SessionEntry | None:
        """Return entry and bump last_seen. Returns None if missing or expired."""
        entry = self._sessions.get(session_id)
        if entry is None:
            return None
        now = time.time()
        if now - entry.last_seen > self._ttl:
            del self._sessions[session_id]
            return None
        entry.last_seen = now
        return entry

    def delete(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def evict_stale(self) -> int:
        """Remove all expired sessions. Return count evicted."""
        now = time.time()
        stale = [
            sid for sid, e in self._sessions.items()
            if now - e.last_seen > self._ttl
        ]
        for sid in stale:
            del self._sessions[sid]
        return len(stale)

    def __len__(self) -> int:
        return len(self._sessions)
