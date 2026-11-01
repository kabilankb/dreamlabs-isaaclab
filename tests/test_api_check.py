import pytest

from dreamlabs_isaaclab import api_check, launcher


@pytest.fixture
def credentials(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "CREDENTIALS", tmp_path / "creds/credentials")
    return tmp_path / "creds/credentials"


def test_no_key_fails_and_says_where_to_add_one(credentials, capsys):
    assert api_check.main([]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_a_refused_key_stops_before_any_model_is_asked(credentials, monkeypatch, capsys):
    launcher.save_key("dl_live_" + "E" * 30, credentials)
    monkeypatch.setattr(launcher, "check_key", lambda key, url: (False, "Dream refused the key (HTTP 401)"))
    monkeypatch.setattr(api_check, "selftest", lambda *a: pytest.fail("must not ask the models of a refused key"))
    assert api_check.main([]) == 1
    out = capsys.readouterr().out
    assert "refused" in out and "E" * 30 not in out


def test_a_passing_run_prints_models_and_only_calls_the_model_when_asked(credentials, monkeypatch, capsys):
    launcher.save_key("dl_live_" + "F" * 30, credentials)
    monkeypatch.setattr(launcher, "check_key", lambda key, url: (True, "Dream accepted the key"))
    calls = []
    monkeypatch.setattr(api_check, "selftest", lambda model, call, py: calls.append(call) or
                        {"models": ["dreamlabs/dream-1-beta"], "rows": 48, "width": 7, "ms": 640.0})
    assert api_check.main([]) == 0
    assert "no model request sent" in capsys.readouterr().out and calls == [False]
    assert api_check.main(["--call-model"]) == 0
    out = capsys.readouterr().out
    assert "48 rows x 7 values in 640 ms" in out and calls == [False, True]
    assert "F" * 30 not in out                                              # the key itself is never printed
