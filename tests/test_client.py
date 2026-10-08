import pytest
from conftest import FakeClock, FakeFinMind, make_client

from twbt.data.client import FinMindError, RateLimiter


def test_rate_limiter_blocks_until_window_frees():
    clock = FakeClock()
    rl = RateLimiter(3, period=3600, clock=clock, sleep=clock.sleep)
    for _ in range(3):
        rl.acquire()
    assert clock.t == 0
    rl.acquire()  # 4th call must wait for the first one to age out
    assert clock.t == pytest.approx(3600)


def test_rate_limiter_sliding_not_fixed_window():
    clock = FakeClock()
    rl = RateLimiter(2, period=100, clock=clock, sleep=clock.sleep)
    rl.acquire()           # t=0
    clock.t = 60
    rl.acquire()           # t=60
    rl.acquire()           # waits until t=100 (first call ages out), not t=160
    assert clock.t == pytest.approx(100)


def test_server_rate_limit_waits_then_retries():
    server = FakeFinMind({"X": [{"date": "2024-01-01", "v": 1}]})
    server.script = [(402, {"status": 402, "msg": "Requests reach the upper limit"})]
    clock = FakeClock()
    client = make_client(server, clock)
    df = client.get("X")
    assert len(df) == 1
    assert clock.t == pytest.approx(3600)  # waited a full window
    assert len(server.requests) == 2


def test_transient_error_backs_off_and_retries():
    server = FakeFinMind({"X": [{"v": 1}]})
    server.script = [(503, {"msg": "down"}), (500, {"msg": "down"})]
    clock = FakeClock()
    client = make_client(server, clock)
    assert len(client.get("X")) == 1
    assert clock.t == pytest.approx(1 + 2)  # 1s then 2s backoff


def test_client_error_is_raised_not_retried():
    server = FakeFinMind({})
    server.fail_on.add(("X", None))
    client = make_client(server)
    with pytest.raises(FinMindError):
        client.get("X")
    assert len(server.requests) == 1


def test_empty_result_is_empty_frame():
    client = make_client(FakeFinMind({}))
    assert client.get("Nothing", data_id="1234").empty
