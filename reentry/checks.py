"""Запуск проверок verify_step*.py без pytest.

Проверки — обычные функции test_*, которые падают через assert. Под pytest
они собираются как есть (см. pytest.ini). Этот модуль даёт тот же прогон
командой `python verify_stepN.py`: вызывает все test_* в порядке объявления,
печатает их вывод, считает провалы и возвращает код выхода.
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
            print(f"   -> ПРОВАЛ в {name}: {exc}\n")
        except Exception:
            failed.append(name)
            print(f"   -> ОШИБКА в {name}:")
            traceback.print_exc()
            print()
    if recorder is not None:
        recorder["passed"] = len(tests) - len(failed)
        recorder["total"] = len(tests)
        recorder.save(script)
    if failed:
        print(f"ИТОГ: провалов {len(failed)} из {len(tests)}: {', '.join(failed)}\n")
        return 1
    print(f"ИТОГ: пройдены все проверки ({len(tests)} из {len(tests)})\n")
    return 0
