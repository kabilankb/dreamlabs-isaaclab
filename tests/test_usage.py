import json
import os
import stat

import pytest

from dreamlabs_isaaclab import usage

KEY_ID = "ab" * 32  # what the export calls key_id: a 64-character identifier


def export_text(period="2026-11", requests=100, sessions=10, minutes=1.0, cost=0.5, key_id=KEY_ID, with_total=True):
    lines = ["period,row,key,key_id,requests,sessions,inference_minutes,estimated_cost_usd",
             f"{period},key,my_key,{key_id},{requests},{sessions},{minutes},{cost}"]
    if with_total:
        lines.append(f"{period},total,,,{requests},{sessions},{minutes},{cost}")
    lines.append(f"{period}-01,day,,,{requests},{sessions},{minutes},")
    return "\n".join(lines) + "\n"


def ep(task="put apple into the bowl", robot="yam", requests=10, success=True, model_s=5.0, run=None, t="2026-11-01T10:00:00+00:00",
       model="m1", episode=0, sessions=1):
    return {"t": t, "kind": "episode", "robot": robot, "instruction": task, "model": model, "run": run, "episode": episode,
            "requests": requests, "sessions": sessions, "model_s": model_s, "success": success}


# ---- the ledger -----------------------------------------------------------------------------------------------------

def test_record_and_read_round_trip_and_skip_damaged_lines(tmp_path):
    ledger = tmp_path / "l.jsonl"
    usage.record({"kind": "episode", "requests": 3}, ledger)
    ledger.write_text(ledger.read_text() + "not json\n{\"no\": \"t\"}\n[1]\n", encoding="utf-8")
    usage.record({"kind": "episode", "requests": 4}, ledger)
    assert [e["requests"] for e in usage.read(ledger)] == [3, 4]


def test_record_never_raises_even_when_the_ledger_cannot_be_written(tmp_path, capsys):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    usage.record({"requests": 1}, blocker / "sub/ledger.jsonl")       # the parent is a file
    assert "could not write" in capsys.readouterr().err


# ---- the rate -------------------------------------------------------------------------------------------------------

def test_pricing_from_file_then_environment_and_unknown_by_default(tmp_path):
    assert not usage.load_pricing(tmp_path / "none.json").known
    f = tmp_path / "p.json"
    f.write_text(json.dumps({"currency": "USD", "per_request": 0.01, "per_session": 0.2}))
    p = usage.load_pricing(f)
    assert (p.currency, p.per_request, p.per_session, p.known) == ("USD", 0.01, 0.2, True)
    assert usage.load_pricing(f, env={"DREAMLABS_RATE_PER_REQUEST": "0.5"}).per_request == 0.5
    assert p.cost(100, 2) == pytest.approx(1.4)
    assert usage.Pricing().cost(100, 2) is None


@pytest.mark.parametrize("raw", ['{"per_request": -1}', '{"per_request": "cheap"}', '{"per_request": NaN}',
                                 '{"per_session": null}', "[1]", "{broken"])
def test_a_bad_rate_is_an_error_not_a_silent_zero(tmp_path, raw):
    f = tmp_path / "p.json"
    f.write_text(raw)
    with pytest.raises(ValueError):
        usage.load_pricing(f)


# ---- Dream's export -------------------------------------------------------------------------------------------------

def test_parse_export_reads_numbers_and_never_the_key_identifier(tmp_path):
    f = tmp_path / "e.csv"
    f.write_text(export_text())
    e = usage.parse_export(f)
    assert e["period"] == "2026-11" and e["keys"] == [{"name": "my_key", "requests": 100, "sessions": 10, "minutes": 1.0, "cost": 0.5}]
    assert e["total"]["requests"] == 100 and e["days"][0]["day"] == "2026-11-01" and e["days"][0]["cost"] is None
    assert KEY_ID not in json.dumps(e)


@pytest.mark.parametrize("text, why", [
    ("a,b\n1,2\n", "not a Dream usage export"),
    (export_text(requests=-5), "zero or more"),
    (export_text(requests="many"), "not a number"),
    (export_text(period="soon"), "no month"),
])
def test_an_export_that_is_not_one_is_refused_with_the_reason(tmp_path, text, why):
    f = tmp_path / "e.csv"
    f.write_text(text)
    with pytest.raises(ValueError, match=why):
        usage.parse_export(f)


def test_a_missing_total_is_summed_and_a_mismatch_is_noted(tmp_path):
    f = tmp_path / "e.csv"
    f.write_text(export_text(with_total=False))
    e = usage.parse_export(f)
    assert e["total"]["requests"] == 100 and "summed" in e["note"]
    f.write_text(export_text().replace("total,,,100", "total,,,999"))
    assert "differs" in usage.parse_export(f)["note"]


def test_import_keeps_a_copy_without_the_key_identifier_mode_600(tmp_path):
    src = tmp_path / "dl.csv"
    src.write_text(export_text())
    kept = usage.import_export(src, tmp_path / "kept")
    text = kept.read_text()
    assert KEY_ID not in text and "key_id" not in text and "my_key" in text
    assert stat.S_IMODE(os.stat(kept).st_mode) == 0o600
    again = usage.import_export(src, tmp_path / "kept")                    # re-importing replaces, never duplicates
    assert again == kept and len(list((tmp_path / "kept").iterdir())) == 1
    assert usage.parse_export(kept)["total"]["requests"] == 100             # and the copy parses as an export itself


def test_the_price_is_dreams_own_cost_per_request(tmp_path):
    f = tmp_path / "e.csv"
    f.write_text(export_text(requests=500, cost=1.0))
    p = usage.pricing_from_exports([usage.parse_export(f)])
    assert p.currency == "USD" and p.per_request == pytest.approx(1.0 / 500) and "2026-11" in p.source
    f.write_text(export_text(requests=500, cost=0))
    assert usage.pricing_from_exports([usage.parse_export(f)]) is None       # nothing billed: no price to derive


def test_your_own_rate_wins_over_the_export(tmp_path, monkeypatch):
    usage.import_export(_write(tmp_path / "e.csv", export_text(requests=100, cost=1.0)))
    assert usage.effective_pricing().per_request == pytest.approx(0.01)
    monkeypatch.setenv("DREAMLABS_RATE_PER_REQUEST", "0.5")
    p = usage.effective_pricing()
    assert p.per_request == 0.5 and "your own" in p.source


def _write(path, text):
    path.write_text(text)
    return path


# ---- the report -----------------------------------------------------------------------------------------------------

def test_summarize_by_task_counts_only_scored_episodes_as_wins_and_prices_the_requests():
    entries = [ep(requests=10, success=True), ep(requests=20, success=False), ep(requests=5, success=None, model_s=None),
               ep(task="other", requests=1)]
    out = usage.summarize(entries, "task", usage.Pricing("USD", 0.01, 0.0, "x"))
    first = out["rows"][0]
    assert first["key"] == "YAM · put apple into the bowl" and first["requests"] == 35 and first["episodes"] == 3
    assert (first["wins"], first["scored"]) == (1, 2) and first["requests_per_episode"] == pytest.approx(11.7, abs=0.05)
    assert first["cost"] == pytest.approx(0.35) and first["model_s"] == 10.0 and first["model_s_partial"] is True
    assert out["total"]["requests"] == 36 and out["total"]["cost"] == pytest.approx(0.36) and out["priced"] is True


def test_no_rate_means_no_cost_not_zero():
    out = usage.summarize([ep()], "task", usage.Pricing())
    assert out["rows"][0]["cost"] is None and out["priced"] is False


def test_groupings_since_and_sessions():
    entries = [ep(robot="rebot", run="r1", model="a", t="2026-10-31T09:00:00+00:00"), ep(run="r1", model="b"),
               {"t": "2026-11-01T11:00:00+00:00", "kind": "test", "robot": "api-test", "instruction": "connection test",
                "model": "b", "requests": 1, "sessions": 1, "model_s": 2.6}]
    assert {r["key"] for r in usage.summarize(entries, "robot")["rows"]} == {"reBot duo", "YAM", "API test"}
    assert {r["key"] for r in usage.summarize(entries, "model")["rows"]} == {"a", "b"}
    assert {r["key"] for r in usage.summarize(entries, "day")["rows"]} == {"2026-10-31", "2026-11-01"}
    assert usage.summarize(entries, "run")["rows"][0]["key"] == "r1"
    assert usage.summarize(entries, "task", since="2026-11-01")["total"]["requests"] == 11
    priced = usage.summarize(entries, "robot", usage.Pricing("USD", 0.0, 0.5, "x"))
    assert priced["total"]["sessions"] == 3 and priced["total"]["cost"] == pytest.approx(1.5)   # a test counts as a session, not an episode
    assert priced["total"]["episodes"] == 2
    with pytest.raises(ValueError):
        usage.summarize(entries, "colour")


def test_reconciliation_against_dreams_totals(tmp_path):
    e = usage.parse_export(_write(tmp_path / "e.csv", export_text(period="2026-11", requests=200, sessions=20, cost=1.0)))
    out = usage.reconcile([ep(requests=50), ep(requests=30, t="2026-10-02T00:00:00+00:00")], [e])
    assert out == [{**out[0], "period": "2026-11", "dream_requests": 200, "ledger_requests": 50, "share": 0.25, "keys": ["my_key"]}]


# ---- importing runs -------------------------------------------------------------------------------------------------

def make_run(root, name, command, log, results=None):
    d = root / name
    d.mkdir(parents=True)
    (d / "command.txt").write_text(command + "\n")
    (d / "run.log").write_text(log)
    if results is not None:
        (d / "results.json").write_text(json.dumps(results))
    return d


def test_backfill_imports_results_and_logs_once(tmp_path):
    runs = tmp_path / "runs"
    make_run(runs, "20261101-100000-yam-apple",
             "/py -m dreamlabs_isaaclab.yam --checkpoint dreamlabs/dream-1-beta --task PutAppleInBowl-v0 --instruction 'put apple into the bowl' --headless",
             "[dream] chunk 9 at step 0\n", [{"episode": 0, "success": True, "steps": 300, "requests": 9, "model_ms_mean": 1000.0}])
    make_run(runs, "20261101-110000-yam-apple",
             "/py -m dreamlabs_isaaclab.yam --checkpoint dreamlabs/dream-1-beta --task PutAppleInBowl-v0 --instruction 'put orange into the bowl'",
             "[dream] chunk 3 at step 90\n[dream] chunk 12 at step 500\n")
    make_run(runs, "20261101-120000-yam-mujoco", "/dream/bin/python /x/dream_control.py --task 'put apple into the bowl' --model dreamlabs/dream-1-beta",
             "  Dream · 0.47s RTT · 30 requests · 1392 actions\n")
    make_run(runs, "20261101-130000-nothing", "/bin/echo hello", "")
    ledger = tmp_path / "l.jsonl"
    assert usage.backfill(runs, ledger) == 3
    assert usage.backfill(runs, ledger) == 0                                     # running it again adds nothing
    by = {e["run"]: e for e in usage.read(ledger)}
    done, stopped, mujoco = by["20261101-100000-yam-apple"], by["20261101-110000-yam-apple"], by["20261101-120000-yam-mujoco"]
    assert (done["requests"], done["success"], done["model_s"], done["instruction"]) == (9, True, 9.0, "put apple into the bowl")
    assert (stopped["requests"], stopped["success"], stopped["model_s"]) == (12, None, None)
    assert mujoco["robot"] == "yam-mujoco" and mujoco["requests"] == 30


def test_import_logs_reads_the_robot_and_instruction_from_the_log(tmp_path):
    root = tmp_path / "scratch"
    (root / "a").mkdir(parents=True)
    (root / "a/run.log").write_text("[dream] policy open: dreamlabs/dream-1-beta (YAM station keys); instruction: 'put apple into the bowl'\n"
                                    "[dream] chunk 7 at step 270\n")
    (root / "b").mkdir()
    (root / "b/run.log").write_text("[dream] policy open: dreamlabs/dream-1-beta on arm model seeed_faas\n[dream] instruction: 'pick up the pen'\n"
                                    "[dream] chunk 4 at step 135\n")
    (root / "c").mkdir()
    (root / "c/run.log").write_text("nothing from Dream here\n")
    ledger = tmp_path / "l.jsonl"
    assert usage.import_logs(root, ledger) == 2 and usage.import_logs(root, ledger) == 0
    got = {e["run"]: e for e in usage.read(ledger)}
    assert got["log:a"]["robot"] == "yam" and got["log:a"]["requests"] == 7 and got["log:a"]["instruction"] == "put apple into the bowl"
    assert got["log:b"]["robot"] == "rebot" and got["log:b"]["instruction"] == "pick up the pen"


def test_parse_log_requests():
    assert usage.parse_log_requests("[dream] chunk 3 at step 1\n[dream] chunk 11 at step 2\n") == 11
    assert usage.parse_log_requests("  Dream · 0.55s RTT · 13 requests · 576 actions\n") == 13
    assert usage.parse_log_requests("nothing") == 0


# ---- the command line -----------------------------------------------------------------------------------------------

def test_cli_prints_the_table_the_rate_source_and_the_reconciliation(tmp_path, capsys):
    src = _write(tmp_path / "dl.csv", export_text(requests=100, cost=1.0))
    usage.record(ep(requests=40, t="2026-11-01T10:00:00+00:00"))
    assert usage.main(["--import-csv", str(src), "--no-import"]) == 0
    out = capsys.readouterr().out
    assert "YAM · put apple into the bowl" in out and "0.4000" in out and "Dream's export for 2026-11" in out
    assert "40 requests (40%)" in out and "not an invoice" in out and KEY_ID not in out


def test_cli_without_a_rate_says_how_to_set_one_and_json_works(tmp_path, capsys):
    usage.record(ep())
    assert usage.main(["--no-import"]) == 0
    assert "No rate set" in capsys.readouterr().out
    assert usage.main(["--no-import", "--json", "--by", "robot", "--rate", "0.1"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["by"] == "robot" and data["total"]["cost"] == pytest.approx(1.0) and data["rate_source"] == "the rate you passed"


def test_cli_refuses_bad_input(tmp_path, capsys):
    assert usage.main(["--since", "yesterday"]) == 2
    usage.PRICING.parent.mkdir(parents=True, exist_ok=True)
    usage.PRICING.write_text("{broken")
    assert usage.main(["--no-import"]) == 2
    usage.PRICING.write_text("{}")
    assert usage.main(["--import-csv", str(tmp_path / "missing.csv")]) == 2
    assert "usage:" in capsys.readouterr().err
