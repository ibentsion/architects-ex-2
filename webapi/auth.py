"""Shared-password gate for the bridge when it is exposed publicly.

The bridge is a localhost tool by default and stays ungated: with ``UI_PASSWORD``
unset every request passes straight through, which is the local-dev topology in
webui/README.md.

Set ``UI_PASSWORD`` and it becomes the whole access control for a public
deployment (cloud/serve_endpoint.sh --full), where the managed URL is reachable
by anyone. That matters more than it looks: the bridge reads repo files by
design -- the corpus, ``eval_results/``, the graded submission -- and every
question spends the *shared* course Token Factory key.

The session is an HMAC-signed cookie rather than server-side state, so it
survives a bridge restart and needs no store. The signing key is derived from
the password, which also means changing the password invalidates every session.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import time

#: How long a login lasts. Long enough to demo, short enough that a leaked
#: cookie is not permanent.
SESSION_TTL_SECONDS = 12 * 60 * 60
COOKIE_NAME = "ui_session"


def password() -> str:
    """The configured shared password, or "" when the gate is disabled."""
    return os.environ.get("UI_PASSWORD", "").strip()


def enabled() -> bool:
    return bool(password())


def _signing_key() -> bytes:
    """Derived from the password so sessions survive a restart without a
    persisted secret, and so changing the password logs everyone out."""
    return hashlib.sha256(("ui-session:" + password()).encode("utf-8")).digest()


def _sign(expiry: int) -> str:
    return hmac.new(_signing_key(), str(expiry).encode("ascii"), hashlib.sha256).hexdigest()


def issue_session(now: float | None = None) -> tuple[str, int]:
    """(cookie value, max_age) for a freshly authenticated visitor."""
    expiry = int((now if now is not None else time.time()) + SESSION_TTL_SECONDS)
    return f"{expiry}.{_sign(expiry)}", SESSION_TTL_SECONDS


def valid_session(cookie: str | None, now: float | None = None) -> bool:
    """True iff ``cookie`` is a signature we issued and has not expired."""
    if not cookie or "." not in cookie:
        return False
    raw_expiry, _, signature = cookie.partition(".")
    try:
        expiry = int(raw_expiry)
    except ValueError:
        return False
    # Signature first, then expiry: never branch on unverified data.
    if not hmac.compare_digest(signature, _sign(expiry)):
        return False
    return expiry > (now if now is not None else time.time())


def password_matches(candidate: str) -> bool:
    """Constant-time comparison — the password is the only thing between a
    public URL and the shared API key."""
    return hmac.compare_digest(candidate.encode("utf-8"), password().encode("utf-8"))


LOGIN_PAGE = """<!doctype html>
<html lang="he" dir="rtl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex, nofollow">
<title>הראל — סוכן תמיכה</title>
<style>
  :root { color-scheme: light; }
  body { margin:0; min-height:100vh; display:flex; align-items:center;
         justify-content:center; background:#f1f5f9;
         font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
  form { background:#fff; border:1px solid #e2e8f0; border-radius:12px;
         padding:28px; width:min(360px, 90vw);
         box-shadow:0 1px 3px rgba(15,23,42,.08); }
  h1 { margin:0 0 4px; font-size:16px; color:#0f172a; }
  p  { margin:0 0 18px; font-size:12px; color:#64748b; line-height:1.5; }
  input { width:100%; box-sizing:border-box; padding:9px 11px; font-size:14px;
          border:1px solid #cbd5e1; border-radius:8px; background:#f8fafc; }
  input:focus { outline:2px solid #6366f1; outline-offset:1px; }
  button { margin-top:12px; width:100%; padding:9px; font-size:14px;
           font-weight:600; color:#fff; background:#4f46e5; border:0;
           border-radius:8px; cursor:pointer; }
  button:hover { background:#4338ca; }
  .err { margin-top:12px; font-size:12px; color:#b91c1c; }
</style>
</head>
<body>
  <form method="post" action="/login">
    <h1>הראל — סוכן תמיכה</h1>
    <p>הדגמה פרטית. יש להזין את הסיסמה שקיבלת.</p>
    <input type="password" name="password" placeholder="סיסמה" autofocus
           autocomplete="current-password" required>
    <button type="submit">כניסה</button>
    __ERROR__
  </form>
</body>
</html>
"""


def login_page(error: str | None = None) -> str:
    return LOGIN_PAGE.replace(
        "__ERROR__", f'<div class="err">{error}</div>' if error else ""
    )
