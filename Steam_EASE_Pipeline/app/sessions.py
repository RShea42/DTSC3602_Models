"""Sign-in sessions, Juan's design: the browser holds only a random ticket; the Steam ID stays on the server.

Juan kept {ticket: (steamid, expires_at)} in one process's memory. On Modal several containers may serve requests and
any of them can restart, so the same mapping lives in a modal.Dict (shared, server-side storage) there.
  SESSION_BACKEND=memory  local runs and tests (Juan's original behaviour)
  SESSION_BACKEND=modal   modal.Dict named by SESSION_DICT (default "steam-ease-sessions")
Expired tickets are refused and deleted when seen; sweep() clears any left behind (run on a schedule on Modal).
"""
import os
import secrets
import time

TTL = 7200                                   # 2 hours, as in Juan's app


class SessionStore:
    def __init__(self, backend=None, name=None):
        backend = backend or os.environ.get("SESSION_BACKEND", "memory")
        if backend == "modal":
            import modal
            self.d = modal.Dict.from_name(name or os.environ.get("SESSION_DICT", "steam-ease-sessions"), create_if_missing=True)
        elif backend == "memory":
            self.d = {}
        else:
            raise ValueError(f"unknown SESSION_BACKEND {backend!r}")

    def create(self, steamid):
        token = secrets.token_urlsafe(24)
        self.d[token] = (steamid, time.time() + TTL)
        return token

    def get(self, token):
        """Steam ID for a live ticket, else None."""
        if not token:
            return None
        v = self.d.get(token)
        if v is None:
            return None
        sid, expires = v
        if expires <= time.time():
            self.delete(token)
            return None
        return sid

    def delete(self, token):
        if token:
            self.d.pop(token, None)

    def sweep(self):
        """Remove expired tickets; returns how many."""
        now = time.time()
        dead = [k for k, (_, exp) in list(self.d.items()) if exp <= now]
        for k in dead:
            self.d.pop(k, None)
        return len(dead)
