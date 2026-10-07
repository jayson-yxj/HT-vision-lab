from __future__ import annotations

from scripts.groq_client import _bounded_retry_delay


def test_long_rate_limit_wait_fails_fast() -> None:
    try:
        _bounded_retry_delay({"Retry-After": "120"}, "rate limited", 0, 30)
    except OSError as error:
        assert "120.0s" in str(error)
        assert "cached results" in str(error)
    else:
        raise AssertionError("long retry delay was accepted")
    assert _bounded_retry_delay({}, "try again in 500ms", 0, 30) == 0.5


if __name__ == "__main__":
    test_long_rate_limit_wait_fails_fast()
    print("PASS: Groq rate-limit waits are bounded")
