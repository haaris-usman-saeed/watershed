from src.make_bundle import build_archive, verify


def test_bundle_is_byte_deterministic_and_verifiable(tmp_path):
    entries = [("b.csv", b"x,y\n1,2\n"), ("a/readme.md", b"hello\n")]
    h1 = build_archive(entries, tmp_path / "one.tar.gz")
    h2 = build_archive(list(reversed(entries)), tmp_path / "two.tar.gz")
    assert h1 == h2
    assert (tmp_path / "one.tar.gz").read_bytes() == (tmp_path / "two.tar.gz").read_bytes()
    ok, n = verify(tmp_path / "one.tar.gz")
    assert ok and n == 2


def test_bundle_verify_detects_tampering(tmp_path):
    build_archive([("a.txt", b"original")], tmp_path / "b.tar.gz")
    p = tmp_path / "b.tar.gz"
    data = bytearray(p.read_bytes())
    data[-30] ^= 0xFF
    p.write_bytes(bytes(data))
    try:
        ok, _ = verify(p)
    except Exception:
        ok = False
    assert not ok
