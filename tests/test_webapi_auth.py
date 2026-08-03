"""The shared-password gate.

This is the entire access control for a public deployment, so the tests are
about what an attacker gets, not just what a happy path returns.
"""
from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from webapi import auth
from webapi.bridge_app import app


@pytest.fixture
def gated(monkeypatch):
    monkeypatch.setenv("UI_PASSWORD", "correct horse")
    return TestClient(app, follow_redirects=False)


@pytest.fixture
def ungated(monkeypatch):
    monkeypatch.delenv("UI_PASSWORD", raising=False)
    return TestClient(app, follow_redirects=False)


# --- disabled by default (the localhost topology) --------------------------- #


def test_no_password_configured_leaves_everything_open(ungated):
    assert not auth.enabled()
    assert ungated.get("/api/datasets").status_code == 200


def test_login_page_is_not_shown_when_there_is_nothing_to_log_into(ungated):
    response = ungated.get("/login")
    assert response.status_code == 303
    assert response.headers["location"] == "/"


# --- gate closed ------------------------------------------------------------ #


def test_api_without_session_is_401_not_a_redirect(gated):
    # An XHR must not be handed an HTML login page.
    response = gated.get("/api/datasets")
    assert response.status_code == 401
    assert response.json()["detail"] == "authentication required"


def test_browser_navigation_without_session_goes_to_login(gated):
    response = gated.get("/")
    assert response.status_code == 303
    assert response.headers["location"] == "/login"


def test_health_check_stays_reachable_unauthenticated(gated):
    assert gated.get("/healthz").status_code == 200


def test_wrong_password_is_rejected(gated):
    response = gated.post("/login", data={"password": "hunter2"})
    assert response.status_code == 401
    assert auth.COOKIE_NAME not in response.cookies


def test_correct_password_issues_an_httponly_session(gated):
    response = gated.post("/login", data={"password": "correct horse"})
    assert response.status_code == 303
    cookie = response.headers["set-cookie"]
    assert "httponly" in cookie.lower()
    assert "samesite=lax" in cookie.lower()


def test_session_unlocks_the_api(gated):
    gated.post("/login", data={"password": "correct horse"})
    assert gated.get("/api/datasets").status_code == 200


# --- forgery and expiry ----------------------------------------------------- #


def test_forged_cookie_is_refused(gated):
    forged = f"{int(time.time()) + 9999}.{'0' * 64}"
    gated.cookies.set(auth.COOKIE_NAME, forged)
    assert gated.get("/api/datasets").status_code == 401


@pytest.mark.parametrize("value", ["", "garbage", "not-a-number.abc", "123", ".", "9999999999."])
def test_malformed_cookies_are_refused(value):
    assert not auth.valid_session(value)


def test_expired_session_is_refused(monkeypatch):
    monkeypatch.setenv("UI_PASSWORD", "pw")
    value, _ = auth.issue_session(now=1000)
    assert auth.valid_session(value, now=1000 + auth.SESSION_TTL_SECONDS - 1)
    assert not auth.valid_session(value, now=1000 + auth.SESSION_TTL_SECONDS + 1)


def test_changing_the_password_invalidates_existing_sessions(monkeypatch):
    monkeypatch.setenv("UI_PASSWORD", "first")
    value, _ = auth.issue_session()
    assert auth.valid_session(value)
    monkeypatch.setenv("UI_PASSWORD", "second")
    assert not auth.valid_session(value)


def test_a_session_signed_for_one_expiry_does_not_validate_another(monkeypatch):
    """Guards against accepting the signature while trusting a swapped expiry."""
    monkeypatch.setenv("UI_PASSWORD", "pw")
    value, _ = auth.issue_session(now=1000)
    _, _, signature = value.partition(".")
    assert not auth.valid_session(f"{int(time.time()) + 99999}.{signature}")


# --- hosted frontend -------------------------------------------------------- #


@pytest.mark.parametrize(
    "path",
    [
        "/%2e%2e/secret.txt",       # encoded ../ — survives client normalization
        "/%2e%2e%2fsecret.txt",
        "/..%2fsecret.txt",
        "/../secret.txt",           # normalized away by the client; must still not leak
        "/a/../../secret.txt",
    ],
)
def test_bundle_traversal_never_serves_a_file_outside_the_bundle(
    path, tmp_path, monkeypatch, ungated
):
    """The property that matters is "no file outside dist/ is ever returned".

    Unencoded `../` is collapsed by the HTTP client before it reaches the app,
    so it lands on the SPA fallback (200, index.html). Percent-encoded forms do
    reach the handler and must be refused outright. Both are asserted here
    because only the pair of them says "cannot leak".
    """
    (tmp_path / "index.html").write_text("<html>app</html>", encoding="utf-8")
    (tmp_path.parent / "secret.txt").write_text("TOPSECRET", encoding="utf-8")
    monkeypatch.setenv("WEBUI_DIST", str(tmp_path))

    response = ungated.get(path)
    assert "TOPSECRET" not in response.text
    assert response.status_code in (200, 400, 404)
    if response.status_code == 200:
        assert "app" in response.text  # the SPA shell, not a file from outside


def test_unknown_path_falls_back_to_the_spa(tmp_path, monkeypatch, ungated):
    (tmp_path / "index.html").write_text("<html>app</html>", encoding="utf-8")
    monkeypatch.setenv("WEBUI_DIST", str(tmp_path))
    response = ungated.get("/some/client/route")
    assert response.status_code == 200
    assert "app" in response.text


def test_frontend_is_gated_too(tmp_path, monkeypatch, gated):
    (tmp_path / "index.html").write_text("<html>app</html>", encoding="utf-8")
    monkeypatch.setenv("WEBUI_DIST", str(tmp_path))
    response = gated.get("/some/client/route")
    assert response.status_code == 303
    assert response.headers["location"] == "/login"
