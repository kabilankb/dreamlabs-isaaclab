"""Test your Dream API setup from the command line: key, models, and (optionally) one real request.

    python -m dreamlabs_isaaclab.api_check                  # key saved? accepted by Dream? which models? (free)
    python -m dreamlabs_isaaclab.api_check --call-model     # plus one real request: uses account credits

Standard library only here; the model list and the request go through the bridge in the Dream environment
(``~/dream-env``). Your key is read from ``~/.dreamlabs/credentials`` by the Dream library and is never printed: only its
last four characters are shown. Exit code 0 only if every check that was run passed.
"""

from __future__ import annotations

import argparse
import sys

from dreamlabs_isaaclab import launcher, usage
from dreamlabs_isaaclab.client import DEFAULT_DREAM_PYTHON, BridgeError, selftest

DEFAULT_MODEL = "dreamlabs/dream-1-beta"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--call-model", action="store_true", help="also send one real request (uses account credits)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"model for --call-model (default {DEFAULT_MODEL})")
    parser.add_argument("--dream-python", default=str(DEFAULT_DREAM_PYTHON))
    args = parser.parse_args(argv)
    failed = 0

    def report(passed: bool, text: str) -> None:
        nonlocal failed
        failed += not passed
        print(f"  {'PASS' if passed else 'FAIL'}  {text}")

    print("Dream API check")
    credentials = launcher.read_credentials()
    key = credentials.get("DREAM_API_KEY", "")
    report(bool(key), f"a key is saved in {launcher.CREDENTIALS} ({launcher.mask(key)})" if key
           else f"no key in {launcher.CREDENTIALS}: add one on the launcher page, or see the setup guide")
    if not key:
        return 1
    accepted, line = launcher.check_key(key, credentials.get("DREAM_API_URL", "https://platform.dreamlabs.ai"))
    report(accepted, line)
    if not accepted:
        return 1
    try:
        answer = selftest(args.model, args.call_model, args.dream_python)
    except (BridgeError, OSError) as error:
        report(False, f"the Dream environment answered: {str(error)[:300]}")
        return 1
    report(True, f"{len(answer['models'])} models on the account: {', '.join(answer['models'])}")
    if args.call_model:
        if answer.get("rows", 0) > 0:
            usage.record({"kind": "test", "robot": "api-test", "instruction": "connection test", "model": args.model,
                          "requests": 1, "sessions": 1, "model_s": round(answer.get("ms", 0) / 1000.0, 2), "source": "api_check"})
        report(answer.get("rows", 0) > 0,
               f"{args.model} answered one request: {answer.get('rows')} rows x {answer.get('width')} values in "
               f"{answer.get('ms', 0):.0f} ms")
    else:
        print("  (no model request sent; add --call-model to test one, it uses account credits)")
    print("All checks passed." if not failed else f"{failed} check(s) failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
