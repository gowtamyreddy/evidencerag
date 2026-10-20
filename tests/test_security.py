"""Security tests for URL ingestion (SSRF) and ZIP upload validation.

Everything here is offline: DNS resolution and HTTP are monkeypatched, so
no test touches a real network, let alone a real internal service. These
tests pin the defenses described in app/main.py — if one starts failing,
a protection regressed.
"""

import io
import sys
import zipfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import app.main as main  # noqa: E402

PDF_BYTES = b"%PDF-1.4\n%fake-but-correctly-signed\n"


def _addrinfo(ip):
    return [(2, 1, 6, "", (ip, 0))]


class _FakeResponse:
    def __init__(self, status_code=200, headers=None, chunks=()):
        self.status_code = status_code
        self.headers = headers or {}
        self._chunks = list(chunks)

    def iter_content(self, chunk_size=65536):
        yield from self._chunks

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(f"HTTP {self.status_code}")

    def close(self):
        pass


@pytest.fixture
def dns(monkeypatch):
    """Map host -> IP for the fake resolver."""
    table = {}

    def fake_getaddrinfo(host, port, proto=0, *args, **kwargs):
        if host not in table:
            import socket

            raise socket.gaierror(f"unknown host {host}")
        return _addrinfo(table[host])

    monkeypatch.setattr(main.socket, "getaddrinfo", fake_getaddrinfo)
    return table


# --- Host validation -------------------------------------------------------


def test_private_and_metadata_hosts_are_blocked(dns):
    dns["internal.example"] = "127.0.0.1"
    dns["metadata.example"] = "169.254.169.254"
    dns["lan.example"] = "192.168.1.10"
    for host in ("internal.example", "metadata.example", "lan.example"):
        with pytest.raises(ValueError):
            main._assert_public_host(host)


def test_public_host_is_allowed(dns):
    dns["papers.example"] = "93.184.216.34"
    main._assert_public_host("papers.example")  # must not raise


def test_mixed_public_and_private_resolution_is_blocked(dns, monkeypatch):
    """A host resolving to BOTH a public and a private address is rejected:
    accepting it would let the connector pick the private one."""

    def fake(host, port, proto=0, *args, **kwargs):
        return _addrinfo("93.184.216.34") + _addrinfo("10.0.0.5")

    monkeypatch.setattr(main.socket, "getaddrinfo", fake)
    with pytest.raises(ValueError):
        main._assert_public_host("mixed.example")


# --- Fetch behavior --------------------------------------------------------


def test_fetch_rejects_non_http_schemes(dns):
    with pytest.raises(ValueError):
        main._fetch_url_safely("file:///etc/passwd")


def test_fetch_follows_public_redirect_and_returns_body(dns, monkeypatch):
    dns["a.example"] = "93.184.216.34"
    dns["b.example"] = "93.184.216.35"
    pages = {
        "http://a.example/paper": _FakeResponse(302, {"Location": "http://b.example/paper.pdf"}),
        "http://b.example/paper.pdf": _FakeResponse(
            200, {"Content-Type": "application/pdf"}, [PDF_BYTES]
        ),
    }
    monkeypatch.setattr(main.requests, "get", lambda url, **kw: pages[url])
    content, content_type = main._fetch_url_safely("http://a.example/paper")
    assert content == PDF_BYTES
    assert content_type == "application/pdf"


def test_fetch_blocks_redirect_to_private_host(dns, monkeypatch):
    """The classic SSRF bypass: a public URL that redirects inward."""
    dns["a.example"] = "93.184.216.34"
    dns["evil-internal"] = "127.0.0.1"
    pages = {
        "http://a.example/go": _FakeResponse(302, {"Location": "http://evil-internal/admin"}),
    }
    monkeypatch.setattr(main.requests, "get", lambda url, **kw: pages[url])
    with pytest.raises(ValueError):
        main._fetch_url_safely("http://a.example/go")


def test_fetch_aborts_oversized_downloads(dns, monkeypatch):
    dns["big.example"] = "93.184.216.34"
    monkeypatch.setattr(main, "MAX_FETCH_BYTES", 1000)
    monkeypatch.setattr(
        main.requests,
        "get",
        lambda url, **kw: _FakeResponse(200, {}, [b"x" * 600, b"y" * 600]),
    )
    with pytest.raises(ValueError):
        main._fetch_url_safely("http://big.example/blob")


# --- ZIP upload validation ---------------------------------------------------


def _zip_bytes(members):
    """members: list of (arcname, bytes)."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for arcname, data in members:
            archive.writestr(arcname, data)
    return buffer.getvalue()


@pytest.fixture
def uploads_dir(tmp_path, monkeypatch):
    target = tmp_path / "uploads"
    target.mkdir()
    monkeypatch.setattr(main, "UPLOADS_DIR", target)
    return target


def test_zip_traversal_is_flattened_into_uploads(uploads_dir):
    paths = main._extract_pdfs(
        _zip_bytes([("../../evil.pdf", PDF_BYTES), ("nested/dir/paper.pdf", PDF_BYTES)])
    )
    names = sorted(p.name for p in paths)
    assert names == ["evil.pdf", "paper.pdf"]
    for path in paths:
        assert path.parent == uploads_dir


def test_zip_rejects_fake_pdfs_and_preserves_existing_corpus(uploads_dir):
    keeper = uploads_dir / "keeper.pdf"
    keeper.write_bytes(PDF_BYTES)
    result = main._extract_pdfs(_zip_bytes([("fake.pdf", b"this is not a pdf at all")]))
    assert result == []
    assert keeper.exists(), "a bad upload must not wipe the working corpus"


def test_zip_caps_pdf_count(uploads_dir):
    members = [(f"paper{i}.pdf", PDF_BYTES) for i in range(55)]
    paths = main._extract_pdfs(_zip_bytes(members))
    assert len(paths) == main.MAX_PDFS == 50


def test_looks_like_pdf():
    assert main._looks_like_pdf(b"%PDF-1.7 ...")
    assert not main._looks_like_pdf(b"<html>not a pdf</html>")
    assert not main._looks_like_pdf(b"")
