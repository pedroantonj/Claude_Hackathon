#!/usr/bin/env python3
"""Minimal runner when pytest is unavailable:  python3 tests/run_tests.py [-k substring] [file ...]"""
import importlib
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _util import Skip  # noqa: E402


def main():
    args = sys.argv[1:]
    k = None
    if "-k" in args:
        i = args.index("-k"); k = args[i + 1]; del args[i:i + 2]
    files = args or sorted(f for f in os.listdir(HERE) if f.startswith("test_") and f.endswith(".py"))
    res = {"PASS": 0, "FAIL": 0, "SKIP": 0, "XFAIL": 0, "XPASS": 0}
    for f in files:
        mod = importlib.import_module(os.path.basename(f)[:-3])
        for name in [n for n in vars(mod) if n.startswith("test_") and callable(getattr(mod, n))]:
            if k and k not in name:
                continue
            fn = getattr(mod, name)
            t0 = time.time()
            xf = getattr(fn, "_xfail", None)
            try:
                fn()
                st = "XPASS" if xf else "PASS"
                msg = ""
            except Skip as e:
                st, msg = "SKIP", str(e)
            except Exception as e:  # noqa: BLE001
                st = "XFAIL" if xf else "FAIL"
                msg = xf if xf else "".join(traceback.format_exception(e)[-3:])
            res[st] += 1
            print(f"{st:5s} {f}::{name} ({time.time() - t0:.1f}s) {msg if st != 'PASS' else ''}".rstrip(), flush=True)
    print(" ".join(f"{k}={v}" for k, v in res.items()))
    sys.exit(1 if res["FAIL"] else 0)


if __name__ == "__main__":
    main()
