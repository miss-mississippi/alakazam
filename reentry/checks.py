"""Running the verify_step*.py checks without pytest.

Checks are plain test_* functions that fail through assert. pytest collects
them as they are (see pytest.ini). This module gives the same run via
`python verify_stepN.py`: it calls every test_* in declaration order, prints
their output, counts failures and returns an exit code.
"""

from __future__ import annotations

import traceback


def run_checks(namespace: dict, recorder=None, script: str | None = None) -> int:
    tests = [(name, obj) for name, obj in namespace.items()
             if name.startswith("test_") and callable(obj)]
    failed = []
    print()
    for name, fn in tests:
        try:
            fn()
        except AssertionError as exc:
            failed.append(name)
            print(f"   -> FAILED in {name}: {exc}\n")
        except Exception:
            failed.append(name)
            print(f"   -> ERROR in {name}:")
            traceback.print_exc()
            print()
    if recorder is not None:
        recorder["passed"] = len(tests) - len(failed)
        recorder["total"] = len(tests)
        recorder.save(script)
    if failed:
        print(f"RESULT: {len(failed)} of {len(tests)} checks failed: {', '.join(failed)}\n")
        return 1
    print(f"RESULT: all checks passed ({len(tests)} of {len(tests)})\n")
    return 0
