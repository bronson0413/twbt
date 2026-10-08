from conftest import make_client

from twbt.cli import main
from twbt.data.store import Manifest, RawStore
from twbt.data.update import Updater


def test_update_without_token_explains(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("FINMIND_TOKEN", raising=False)
    assert main(["--data-dir", str(tmp_path), "update"]) == 2
    assert "FINMIND_TOKEN" in capsys.readouterr().err


def test_build_and_status(server, tmp_path, fixed_now, capsys):
    Updater(make_client(server), RawStore(tmp_path), Manifest(tmp_path),
            end="2026-10-08", now=lambda: fixed_now["now"]).run()
    assert main(["--data-dir", str(tmp_path), "build"]) == 0
    assert main(["--data-dir", str(tmp_path), "status"]) == 0
    out = capsys.readouterr().out
    assert "universe: 3 stocks (1 delisted)" in out
    assert "TaiwanStockPrice" in out and "3/3" in out
