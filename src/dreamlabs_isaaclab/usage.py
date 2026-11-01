"""API usage and estimated spend, per task: what your runs asked of the Dream API and what that costs at *your* rate.

    python -m dreamlabs_isaaclab.usage                  # per task (the default)
    python -m dreamlabs_isaaclab.usage --by run         # or: task, robot, model, run, day
    python -m dreamlabs_isaaclab.usage --since 2026-11-01 --json

What is measured: every request to Dream's model (one ``get_action`` call each), the sessions (one per episode, one per
connection test with a real request), the time spent waiting for answers, and whether the episode succeeded. The sim
clients write one line per episode to a ledger, ``~/.dreamlabs-isaaclab/usage.jsonl``; earlier launcher runs are imported
from ``~/.dreamlabs-isaaclab/runs/`` the first time.

What is **not** known: Dream's library has no usage, balance or price call, so the money is an estimate: your rate times the
measured usage. Put your rate in ``~/.dreamlabs-isaaclab/pricing.json`` (or ``DREAMLABS_RATE_PER_REQUEST`` /
``DREAMLABS_RATE_PER_SESSION`` / ``DREAMLABS_CURRENCY``)::

    {"currency": "credits", "per_request": 0.05, "per_session": 0.0}

With no rate the cost column shows "-". Your Dream account page is the authority on what you were charged.

Better than a guess: import the usage export from your Dream account (a CSV with one row per key, a total and one per day)::

    python -m dreamlabs_isaaclab.usage --import-csv ~/Downloads/dreamlabs-usage-2026-11.csv

The cost per request is then Dream's own (its estimated cost divided by its requests that month), each task gets its share, and a
reconciliation shows how much of Dream's total the ledger can attribute to a task. The export holds a key identifier; the
imported copy keeps only the key's name.

Standard library only; it is imported by the simulator clients, so it must stay cheap and safe to call.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shlex
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

HOME = Path.home()
LEDGER = Path(os.environ.get("DREAMLABS_USAGE_LEDGER") or HOME / ".dreamlabs-isaaclab/usage.jsonl")
PRICING = HOME / ".dreamlabs-isaaclab/pricing.json"
RUNS = HOME / ".dreamlabs-isaaclab/runs"
EXPORTS = HOME / ".dreamlabs-isaaclab/dream-usage"
GROUPS = ("task", "robot", "model", "run", "day")

ROBOT_NAMES = {"rebot": "reBot duo", "yam": "YAM", "yam-mujoco": "YAM (Dream's MuJoCo)", "api-test": "API test"}


# ---- the ledger -----------------------------------------------------------------------------------------------------


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def record(entry: dict, ledger: Path | None = None) -> None:
    """Append one usage line. Never raises: losing a usage line must not stop a simulation."""
    path = Path(ledger) if ledger else LEDGER
    line = {"t": now(), **entry}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(line, separators=(",", ":")) + "\n")
    except OSError as error:
        print(f"[usage] could not write {path}: {error}", file=sys.stderr)


def read(ledger: Path | None = None) -> list[dict]:
    """Every well-formed line of the ledger; a damaged line is skipped."""
    path = Path(ledger) if ledger else LEDGER
    entries = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
            except ValueError:
                continue
            if isinstance(item, dict) and "t" in item:
                entries.append(item)
    except OSError:
        pass
    return entries


# ---- the rate -------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Pricing:
    """Your rate. Dream publishes no usage API, so this is only ever an estimate."""

    currency: str = "credits"
    per_request: float = 0.0
    per_session: float = 0.0
    source: str = ""

    @property
    def known(self) -> bool:
        return self.per_request > 0 or self.per_session > 0

    def cost(self, requests: int, sessions: int) -> float | None:
        return round(requests * self.per_request + sessions * self.per_session, 6) if self.known else None


def _rate(value: object, name: str) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number") from None
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be zero or more")
    return number


def load_pricing(path: Path | None = None, env: dict[str, str] | None = None) -> Pricing:
    """Your rate from the environment, else ``pricing.json``, else none. A bad value is an error, not a silent zero."""
    env = os.environ if env is None else env
    data: dict = {}
    file = Path(path) if path else PRICING
    if file.is_file():
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ValueError(f"{file} is not valid JSON: {error}") from None
        if not isinstance(data, dict):
            raise ValueError(f"{file} must hold a JSON object")
    per_request = env.get("DREAMLABS_RATE_PER_REQUEST", data.get("per_request", 0))
    per_session = env.get("DREAMLABS_RATE_PER_SESSION", data.get("per_session", 0))
    currency = str(env.get("DREAMLABS_CURRENCY", data.get("currency", "credits")))[:12] or "credits"
    return Pricing(currency, _rate(per_request, "per_request"), _rate(per_session, "per_session"))


# ---- Dream's own usage export ---------------------------------------------------------------------------------------

EXPORT_COLUMNS = ("period", "row", "key", "requests", "sessions", "inference_minutes", "estimated_cost_usd")


def _number(text: str | None, name: str, line: int) -> float | None:
    if text is None or text.strip() == "":
        return None
    try:
        value = float(text)
    except ValueError:
        raise ValueError(f"line {line}: {name} is not a number: {text[:20]!r}") from None
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"line {line}: {name} must be zero or more")
    return value


def parse_export(path: Path) -> dict:
    """Dream's usage export as data: ``{"period", "keys": [...], "total", "days": [...], "note"}``.

    Rows are ``key`` (one per API key), ``total`` and ``day``. The key identifier column is never read. Raises ValueError for
    a file that is not this export.
    """
    try:
        handle = open(path, newline="", encoding="utf-8-sig")  # noqa: SIM115 - closed below
    except OSError as error:
        raise ValueError(f"cannot read {path}: {error}") from None
    with handle:
        reader = csv.DictReader(handle)
        missing = [c for c in EXPORT_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path.name} is not a Dream usage export: missing column(s) {', '.join(missing)}")
        keys, days, total, period = [], [], None, ""
        for line, row in enumerate(reader, start=2):
            item = {"requests": int(_number(row["requests"], "requests", line) or 0),
                    "sessions": int(_number(row["sessions"], "sessions", line) or 0),
                    "minutes": _number(row["inference_minutes"], "inference_minutes", line),
                    "cost": _number(row["estimated_cost_usd"], "estimated_cost_usd", line)}
            kind = (row["row"] or "").strip().lower()
            if kind == "key":
                keys.append({"name": (row["key"] or "?")[:64], **item})
                period = period or (row["period"] or "")[:7]
            elif kind == "total":
                total, period = item, (row["period"] or "")[:7] or period
            elif kind == "day":
                days.append({"day": (row["period"] or "")[:10], **item})
    if not re.fullmatch(r"\d{4}-\d{2}", period):
        raise ValueError(f"{path.name}: no month found (expected rows like 2026-11)")
    summed = {"requests": sum(k["requests"] for k in keys), "sessions": sum(k["sessions"] for k in keys),
              "minutes": sum(k["minutes"] or 0 for k in keys), "cost": sum(k["cost"] or 0 for k in keys)}
    note = ""
    if total is None:
        total, note = summed, "no total row: summed from the key rows"
    elif total["requests"] != summed["requests"] and keys:
        note = f"the total row ({total['requests']} requests) differs from the sum of the key rows ({summed['requests']})"
    return {"period": period, "keys": keys, "total": total, "days": days, "note": note}


def import_export(path: Path, directory: Path | None = None) -> Path:
    """Validate Dream's export and keep a copy without the key identifiers, mode 600. Returns the copy's path."""
    parsed = parse_export(Path(path).expanduser())
    target_dir = Path(directory) if directory else EXPORTS
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"dreamlabs-usage-{parsed['period']}.csv"
    rows = [("period", "row", "key", "requests", "sessions", "inference_minutes", "estimated_cost_usd")]
    fmt = lambda v: "" if v is None else f"{v:g}"
    for item in parsed["keys"]:
        rows.append((parsed["period"], "key", item["name"], item["requests"], item["sessions"], fmt(item["minutes"]), fmt(item["cost"])))
    t = parsed["total"]
    rows.append((parsed["period"], "total", "", t["requests"], t["sessions"], fmt(t["minutes"]), fmt(t["cost"])))
    for item in parsed["days"]:
        rows.append((item["day"], "day", "", item["requests"], item["sessions"], fmt(item["minutes"]), fmt(item["cost"])))
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", newline="", encoding="utf-8") as out:
        csv.writer(out).writerows(rows)
    return target


def load_exports(directory: Path | None = None) -> list[dict]:
    """Every imported export, oldest month first; one that no longer parses is skipped."""
    folder = Path(directory) if directory else EXPORTS
    found = []
    for path in sorted(folder.glob("dreamlabs-usage-*.csv")) if folder.is_dir() else []:
        try:
            found.append(parse_export(path))
        except ValueError:
            continue
    return sorted(found, key=lambda e: e["period"])


def pricing_from_exports(exports: list[dict]) -> Pricing | None:
    """Dream's own blended price: the latest month's estimated cost divided by its requests."""
    for export in reversed(exports):
        total = export["total"]
        if total["requests"] > 0 and total["cost"] is not None and total["cost"] > 0:
            return Pricing("USD", total["cost"] / total["requests"], 0.0, f"Dream's export for {export['period']}")
    return None


def effective_pricing(exports: list[dict] | None = None) -> Pricing:
    """Your own rate if you set one, else Dream's from an imported export, else none."""
    own = load_pricing()
    if own.known:
        return Pricing(own.currency, own.per_request, own.per_session, "your own rate (pricing.json or environment)")
    return pricing_from_exports(load_exports() if exports is None else exports) or own


def reconcile(entries: list[dict], exports: list[dict]) -> list[dict]:
    """Dream's totals against what the ledger can attribute to a task, month by month."""
    out = []
    for export in exports:
        month = export["period"]
        mine = [e for e in entries if str(e.get("t", ""))[:7] == month]
        ledger_requests = sum(int(e.get("requests") or 0) for e in mine)
        total = export["total"]
        out.append({"period": month, "dream_requests": total["requests"], "dream_sessions": total["sessions"],
                    "dream_minutes": total["minutes"], "dream_cost": total["cost"], "keys": [k["name"] for k in export["keys"]],
                    "ledger_requests": ledger_requests, "ledger_sessions": sum(int(e.get("sessions") or 0) for e in mine),
                    "share": round(ledger_requests / total["requests"], 3) if total["requests"] else None,
                    "note": export["note"]})
    return out


# ---- importing earlier runs -----------------------------------------------------------------------------------------

_CHUNK = re.compile(r"\[dream\] chunk (\d+) at step")
_MUJOCO = re.compile(r"Dream · [\d.]+s RTT · (\d+) requests")
_MODULES = {"dreamlabs_isaaclab.rebot": "rebot", "dreamlabs_isaaclab.yam": "yam"}


def parse_log_requests(text: str) -> int:
    """How many requests a run log shows: the last ``[dream] chunk N`` (sim clients) or ``N requests`` (Dream's MuJoCo)."""
    sim = [int(n) for n in _CHUNK.findall(text)]
    mujoco = [int(n) for n in _MUJOCO.findall(text)]
    return max(sim + mujoco, default=0)


def _command(directory: Path) -> dict:
    """robot, model, instruction and scene from a run folder's ``command.txt``."""
    try:
        argv = shlex.split((directory / "command.txt").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}

    def option(name: str) -> str | None:
        return argv[argv.index(name) + 1] if name in argv and argv.index(name) + 1 < len(argv) else None

    robot = next((v for k, v in _MODULES.items() if k in argv), None)
    if robot is None and any("dream_control" in part for part in argv):
        robot = "yam-mujoco"
    return {"robot": robot, "model": option("--checkpoint") or option("--model"),
            "instruction": option("--instruction") or option("--task"), "scene": option("--task")}


_POLICY = re.compile(r"\[dream\] policy open: (\S+)(?: on arm model (\S+)| \(YAM station keys\); instruction: '(.*)')")
_INSTRUCTION = re.compile(r"\[dream\] instruction: '(.*)'")


def _from_log(directory: Path) -> dict:
    """Robot, model and (for the YAM client, which logs it) the instruction, from a run log when there is no command.txt."""
    try:
        text = (directory / "run.log").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    match = _POLICY.search(text)
    if not match:
        return {}
    model, arm, instruction = match.groups()
    logged = _INSTRUCTION.search(text)  # the reBot client logs its instruction on a line of its own
    return {"robot": "rebot" if arm else "yam", "model": model,
            "instruction": instruction or (logged.group(1) if logged else "(instruction not logged)")}


def import_folder(directory: Path, run_id: str, stamp: str, ledger: Path | None = None, source: str = "imported") -> int:
    """Record one run folder (``results.json`` per episode, else the request count in ``run.log``). Returns lines added."""
    meta = {**_from_log(directory), **{k: v for k, v in _command(directory).items() if v}}
    if not meta.get("robot"):
        return 0
    base = {"t": stamp, "run": run_id, "source": source, **{k: v for k, v in meta.items() if v}}
    results = directory / "results.json"
    episodes = None
    if results.is_file():
        try:
            episodes = json.loads(results.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            episodes = None
    added = 0
    if isinstance(episodes, list) and episodes:
        for item in episodes:
            requests = int(item.get("requests") or 0)
            model_s = requests * float(item.get("model_ms_mean") or 0.0) / 1000.0
            record({**base, "kind": "episode", "episode": item.get("episode"), "requests": requests, "sessions": 1,
                    "model_s": round(model_s, 2), "success": item.get("success"), "steps": item.get("steps")}, ledger)
            added += 1
    else:  # stopped before an episode finished (or Dream's own sim): all the log can say is the request count
        try:
            requests = parse_log_requests((directory / "run.log").read_text(encoding="utf-8", errors="replace"))
        except OSError:
            requests = 0
        if requests:
            record({**base, "kind": "episode", "episode": None, "requests": requests, "sessions": 1, "model_s": None,
                    "success": None, "note": "stopped or not scored"}, ledger)
            added += 1
    return added


def backfill(runs_dir: Path | None = None, ledger: Path | None = None) -> int:
    """Import launcher run folders that the ledger does not know yet. Returns the lines added."""
    runs = Path(runs_dir) if runs_dir else RUNS
    known = {item.get("run") for item in read(ledger) if item.get("run")}
    added = 0
    for directory in sorted(p for p in runs.glob("*") if p.is_dir()) if runs.is_dir() else []:
        if directory.name in known:
            continue
        try:
            stamp = datetime.strptime(directory.name[:15], "%Y%m%d-%H%M%S").astimezone(timezone.utc).isoformat(timespec="seconds")
        except ValueError:
            stamp = now()
        added += import_folder(directory, directory.name, stamp, ledger)
    return added


def import_logs(root: Path, ledger: Path | None = None) -> int:
    """Import every folder under ``root`` that holds a ``run.log`` (runs started outside the launcher). Idempotent."""
    known = {item.get("run") for item in read(ledger) if item.get("run")}
    added = 0
    for log in sorted(Path(root).expanduser().rglob("run.log")):
        directory = log.parent
        run_id = f"log:{directory.name}"
        if run_id in known:
            continue
        stamp = datetime.fromtimestamp(log.stat().st_mtime, tz=timezone.utc).isoformat(timespec="seconds")
        added += import_folder(directory, run_id, stamp, ledger, source="imported-log")
    return added


# ---- the report -----------------------------------------------------------------------------------------------------


def task_label(entry: dict) -> str:
    robot = ROBOT_NAMES.get(entry.get("robot", ""), entry.get("robot") or "?")
    return f"{robot} · {entry.get('instruction') or entry.get('scene') or 'unknown task'}"


def _key(entry: dict, by: str) -> str:
    if by == "task":
        return task_label(entry)
    if by == "robot":
        return ROBOT_NAMES.get(entry.get("robot", ""), entry.get("robot") or "?")
    if by == "model":
        return entry.get("model") or "?"
    if by == "run":
        return entry.get("run") or f"{entry.get('t', '')[:19]} (not from the launcher)"
    return str(entry.get("t", ""))[:10] or "?"


def summarize(entries: list[dict], by: str = "task", pricing: Pricing | None = None, since: str | None = None) -> dict:
    """Group the ledger. Returns ``{"by", "rows": [...], "total": {...}, "currency", "priced"}``."""
    if by not in GROUPS:
        raise ValueError(f"by must be one of {GROUPS}")
    pricing = pricing or Pricing()
    buckets: dict[str, dict] = defaultdict(lambda: {"requests": 0, "sessions": 0, "episodes": 0, "model_s": 0.0, "wins": 0,
                                                    "scored": 0, "runs": set(), "last": "", "timed": 0, "untimed": 0})
    for entry in entries:
        if since and str(entry.get("t", ""))[:10] < since:
            continue
        bucket = buckets[_key(entry, by)]
        bucket["requests"] += int(entry.get("requests") or 0)
        bucket["sessions"] += int(entry.get("sessions") or 0)
        if entry.get("model_s") is None:
            bucket["untimed"] += 1
        else:
            bucket["timed"] += 1
            bucket["model_s"] += float(entry["model_s"])
        if entry.get("kind") == "episode":
            bucket["episodes"] += 1
        if entry.get("success") is not None:
            bucket["scored"] += 1
            bucket["wins"] += bool(entry.get("success"))
        bucket["runs"].add(entry.get("run") or entry.get("t"))
        bucket["last"] = max(bucket["last"], str(entry.get("t", "")))

    def finish(name: str, b: dict) -> dict:
        episodes = b["episodes"]
        return {"key": name, "runs": len(b["runs"]), "episodes": episodes, "requests": b["requests"], "sessions": b["sessions"],
                "requests_per_episode": round(b["requests"] / episodes, 1) if episodes else None,
                "model_s": round(b["model_s"], 1) if b["timed"] else None, "model_s_partial": bool(b["timed"] and b["untimed"]),
                "wins": b["wins"], "scored": b["scored"],
                "cost": pricing.cost(b["requests"], b["sessions"]), "last": b["last"]}

    rows = sorted((finish(k, v) for k, v in buckets.items()), key=lambda r: (-r["requests"], r["key"]))
    total_bucket = {"requests": sum(r["requests"] for r in rows), "sessions": sum(r["sessions"] for r in rows),
                    "episodes": sum(r["episodes"] for r in rows), "model_s": None, "model_s_partial": False,
                    "timed": sum(b["timed"] for b in buckets.values()), "untimed": sum(b["untimed"] for b in buckets.values()),
                    "_model_s": sum(b["model_s"] for b in buckets.values()),
                    "wins": sum(r["wins"] for r in rows), "scored": sum(r["scored"] for r in rows),
                    "runs": {x for b in buckets.values() for x in b["runs"]}, "last": max((r["last"] for r in rows), default="")}
    total_bucket["model_s"] = total_bucket.pop("_model_s")
    return {"by": by, "rows": rows, "total": finish("total", total_bucket), "currency": pricing.currency, "priced": pricing.known}


def format_table(summary: dict) -> str:
    """The summary as a plain-text table."""
    money = summary["currency"]
    head = ["Task" if summary["by"] == "task" else summary["by"].capitalize(), "Runs", "Episodes", "Requests", "Req/ep",
            "Model s", "Success", f"Est. {money}"]
    lines = [head]
    for row in summary["rows"] + [summary["total"]]:
        lines.append([row["key"], str(row["runs"]), str(row["episodes"]), str(row["requests"]),
                      "-" if row["requests_per_episode"] is None else f"{row['requests_per_episode']:g}",
                      ("-" if row["model_s"] is None else f"{row['model_s']:g}" + ("+" if row["model_s_partial"] else "")),
                      f"{row['wins']}/{row['scored']}" if row["scored"] else "-",
                      "-" if row["cost"] is None else f"{row['cost']:.4f}"])
    widths = [max(len(line[i]) for line in lines) for i in range(len(head))]
    out = []
    for n, line in enumerate(lines):
        out.append("  ".join(cell.ljust(widths[0]) if i == 0 else cell.rjust(widths[i]) for i, cell in enumerate(line)))
        if n == 0 or n == len(lines) - 2:
            out.append("  ".join("-" * w for w in widths))
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--by", choices=GROUPS, default="task", help="how to group (default: task)")
    parser.add_argument("--since", help="only entries on or after this date, YYYY-MM-DD")
    parser.add_argument("--rate", type=float, help="your cost per request (overrides pricing.json)")
    parser.add_argument("--session-rate", type=float, help="your cost per session, i.e. per episode")
    parser.add_argument("--currency", help="label for the cost column (default: credits)")
    parser.add_argument("--json", action="store_true", help="print JSON")
    parser.add_argument("--no-import", action="store_true", help="do not import earlier launcher runs")
    parser.add_argument("--import-csv", metavar="FILE", help="import Dream's usage export (CSV) and use its price")
    parser.add_argument("--import-logs", metavar="DIR", help="import runs started outside the launcher: every folder under DIR with a run.log")
    args = parser.parse_args(argv)
    try:
        if args.import_csv:
            kept = import_export(Path(args.import_csv))
            print(f"Imported Dream's export as {kept} (key identifiers dropped, mode 600).\n")
        if args.import_logs:
            print(f"Imported {import_logs(Path(args.import_logs))} line(s) from run logs under {args.import_logs}.\n")
        pricing = effective_pricing()
        if args.rate is not None or args.session_rate is not None or args.currency:
            pricing = Pricing(args.currency or pricing.currency, _rate(args.rate, "--rate") if args.rate is not None else pricing.per_request,
                              _rate(args.session_rate, "--session-rate") if args.session_rate is not None else pricing.per_session,
                              "the rate you passed")
    except ValueError as error:
        print(f"usage: {error}", file=sys.stderr)
        return 2
    if args.since and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", args.since):
        print("usage: --since is a date like 2026-11-01", file=sys.stderr)
        return 2
    imported = 0 if args.no_import else backfill()
    entries = read()
    summary = summarize(entries, args.by, pricing, args.since)
    summary["reconciliation"] = reconcile(entries, load_exports())
    summary["rate_source"] = pricing.source
    if args.json:
        print(json.dumps(summary, indent=2))
        return 0
    if imported:
        print(f"(imported {imported} line(s) from earlier runs in {RUNS})\n")
    if not summary["rows"]:
        print("No usage recorded yet. Runs from the launcher page and the sim clients are recorded as they finish.")
        return 0
    print(format_table(summary))
    print()
    if summary["priced"]:
        print(f"Estimated cost = requests x {pricing.per_request:.5g} + sessions x {pricing.per_session:g} {pricing.currency}, using "
              f"{pricing.source or 'the rate you set'}. It is an estimate, not an invoice: Dream's account page shows what you were charged.")
    else:
        print(f"No rate set, so cost shows '-'. Dream's library has no usage or price call: put your rate in {PRICING} "
              f'(for example {{"currency": "credits", "per_request": 0.05}}) or pass --rate.')
    for item in summary["reconciliation"]:
        share = "-" if item["share"] is None else f"{item['share'] * 100:.0f}%"
        billed = "" if item["dream_cost"] is None else f", ${item['dream_cost']:.2f}"
        print(f"\nDream's export for {item['period']}: {item['dream_requests']} requests, {item['dream_sessions']} sessions"
              f"{billed}. The ledger attributes {item['ledger_requests']} requests ({share}) to tasks; the rest ran before "
              f"recording started or outside this tool.")
        if item["note"]:
            print(f"  note: {item['note']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
