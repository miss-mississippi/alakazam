# alakazam — вброс алюминия в мезосферу при входе спутников

[![tests](https://github.com/miss-mississippi/alakazam/actions/workflows/tests.yml/badge.svg)](https://github.com/miss-mississippi/alakazam/actions/workflows/tests.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)

Модель входа и абляции спутника, которая считает, **сколько алюминия испаряется
при входе в атмосферу и на каких высотах он вбрасывается**. Штатные инструменты
анализа демиза (DRAMA/SESAM, ORSAT) эту величину не дают: они проектировались под
риск для людей на земле, и их критерий демиза — расплав. Атмосферной химии нужен пар.

> **In English.** A reentry and ablation model of a satellite that estimates how much
> aluminium is vaporized during atmospheric entry and at which altitudes it is
> injected. Standard demise tools (DRAMA/SESAM, ORSAT) treat melting as demise, while
> atmospheric chemistry needs the vaporized mass. Full report (in Russian):
> [docs/REPORT.md](docs/REPORT.md).

Полный отчёт с выводом каждой цифры, допущениями, константами и журналом ошибок —
**[docs/REPORT.md](docs/REPORT.md)**.

## Главный результат

Объект 175 кг, 30% алюминия (52.5 кг Al), распределённого по фрагментам равномерно.
Вход 120 км / 7500 м/с / −1.5°, наклонение 53°, атмосфера NRLMSIS 2.1.

![Высотное распределение вброса алюминия](figures/step5_main.png)

| сценарий поверхности | ε при ~2000 K | испарено Al | выход | медиана вброса | расплав Al |
|---|---|---|---|---|---|
| оксидная плёнка оптически активна | 0.32<!--=step5.scenarios.film.eps2000:.2f--> (0.30<!--=step5.film_shape.eps2000_min:.2f-->–0.33<!--=step5.film_shape.eps2000_max:.2f-->) | **8.5<!--=step5.scenarios.film.total:.1f--> кг** | 16<!--=step5.scenarios.film.yield_pct:.0f-->% | 76.1<!--=step5.scenarios.film.median:.1f--> км | 32.4<!--=step5.scenarios.film.melt:.1f--> кг |
| плёнки нет, голый расплав | ≈0.17<!--=step5.scenarios.bare.eps2000:.2f--> (оценка) | **12.0<!--=step5.scenarios.bare.total:.1f--> кг** | 23<!--=step5.scenarios.bare.yield_pct:.0f-->% | 75.9<!--=step5.scenarios.bare.median:.1f--> км | 32.8<!--=step5.scenarios.bare.melt:.1f--> кг |

Весь вброс идёт выше стратопаузы (50 км), около 10<!--=step5.scenarios.film.above_meso_pct:.0f-->% — выше мезопаузы (85 км).
Расплав — верхняя граница: всё, что в принципе может стать оксидом. Между расплавом
и паром — судьба капельной фазы, которую не разрешает ни одна текущая модель.

## Что показывает работа

1. **Демиз в DRAMA/ORSAT — это расплав, а не испарение.** До полного расплава
   нужно 1.01<!--=step4.criteria.melt:.2f--> МДж/кг (ORSAT: 0.93, расхождение 8<!--=verify_step4.orsat.diff_pct:.0f-->% — свойства сплава),
   до полного испарения 13.6<!--=step4.criteria.vapour:.1f--> МДж/кг, в 13.4<!--=step4.criteria.ratio:.1f--> раза больше.
2. **Алюминий кипит при давлении торможения, а не при 1 атм.** На 70–80 км это
   0.1–1 кПа, и Al кипит при 1759<!--=step4.fragments.film.panels.Tb_min:.0f-->–2039<!--=step4.fragments.film.mli.Tb_max:.0f--> K вместо 2792 K. С кипением при 1 атм
   в сценарии с плёнкой испарилось бы 0.6<!--=step5.changes.film_1atm:.1f--> кг вместо 8.5<!--=step5.scenarios.film.total:.1f-->.
3. **Состояние поверхности — не главная неопределённость.** Сценарии «плёнка» и
   «голый расплав» различаются в 1.4<!--=step5.scenarios.ratio:.1f--> раза: при ~2000 K излучательная способность
   плёнки пришпилена справочной точкой α-Al₂O₃, а у жидкого металла она уже ~0.17.
4. **Массу определяет, где лежит алюминий.** При той же доле 30% распределение Al
   по фрагментам даёт 0<!--=step5.sensitivity.al_split.lo:.0f-->–17<!--=step5.sensitivity.al_split.hi:.0f--> кг, доля тонкостенной массы 25–75% — 4.3<!--=step5.sensitivity.thin_fraction.lo:.1f-->–13.1<!--=step5.sensitivity.thin_fraction.hi:.1f--> кг,
   высота разрушения ±10 км — 4.4<!--=step5.sensitivity.breakup.lo:.1f-->–11.1<!--=step5.sensitivity.breakup.hi:.1f--> кг.
5. **Высота вброса наследуется от высоты разрушения H.** В окне 68–83 км медиана
   вброса ≈ 75.9<!--=step5b.transfer.intercept78:.1f--> + 0.82<!--=step5b.transfer.gain_mid:.2f-->·(H − 78) км; при фиксированной H смещение
   +1.7<!--=step5b.offsets.min:+.1f-->…+2.8<!--=step5b.offsets.max:+.1f--> км по всем остальным параметрам.
6. **Сверка с Ferreira et al. 2024 — согласие по порядку величины.** У них окисляется
   32<!--=step5.ferreira.theirs.pct:.0f-->% Al (молекулярная динамика), у нас испаряется 16<!--=step5.ferreira.film_uniform.pct:.0f-->–23<!--=step5.ferreira.bare_uniform.pct:.0f-->% при равномерном Al и
   32<!--=step5.ferreira.film_thin.pct:.0f-->–46<!--=step5.ferreira.bare_thin.pct:.0f-->%, если Al сосредоточен в тонкостенных элементах.
7. **Эксперимент.** Излучательную способность окисленного Al можно получить из
   отражения при комнатной температуре, но нужен средний ИК в полной полусферической
   геометрии (золотая интегрирующая сфера): без него ошибка −84<!--=step5b.no_midir.T2000.err_pct:.0f-->%.

## Быстрый старт

```bash
git clone https://github.com/miss-mississippi/alakazam.git
cd alakazam
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

python -m pytest                     # все проверки + сверка документов с кодом
```

Пересчитать всё целиком (≈ 3 минуты):

```bash
python run_step1.py && python explore_step1b.py && python run_step2.py
python run_step3.py && python run_step4.py && python run_step5.py
python analysis_step5b.py
for n in 1 2 3 4 5; do python verify_step$n.py; done
python check_docs.py --fix           # обновить числа в документах из results/
```

## Как устроен расчёт

| шаг | скрипт | что делает | график |
|---|---|---|---|
| 1 | `run_step1.py` | 3-DOF траектория, экспоненциальная атмосфера, сверка с Алленом–Эггерсом | [траектория](figures/step1_trajectory.png) |
| 1b | `explore_step1b.py` | пик нагрева против торможения, фрагментация | [фрагментация](figures/step1b_fragmentation.png) |
| 2 | `run_step2.py` | NRLMSIS 2.1, солнечная активность, широта, сезон | [атмосфера](figures/step2_atmosphere.png) |
| 3 | `run_step3.py` | нагрев Саттона–Грейвса, вращение атмосферы, бюджет неопределённостей | [нагрев](figures/step3_heating.png) |
| 4 | `run_step4.py` | фрагменты, поэлементная тепловая модель, кипение при местном давлении | [ε и режимы](figures/step4_epsilon.png) |
| 5 | `run_step5.py` | высотное распределение вброса, бюджет, сверка с Ferreira | [результат](figures/step5_main.png) |
| 5b | `analysis_step5b.py` | передаточная функция высоты, ε(T) по α-Al₂O₃, приборы | [ревизия](figures/step5b_revision.png) |

Физика собрана в пакете `reentry/`: атмосфера, траектория, нагрев, абляция,
излучательная способность.

## Проверки и сверка цифр

- **37<!--=checks.total:.0f--> проверок** в `verify_step1.py` … `verify_step5.py`: каждая бьёт по тождеству
  или независимому бенчмарку (кеплеров тест, энергобаланс внутри ОДУ, Stardust,
  таблица давления паров CRC, аналитическое равновесие пластины и т.д.) и падает
  через `assert`. Запускаются через pytest или по одной: `python verify_step4.py`.
- **Цифры в документах не расходятся с кодом молча.** Скрипты пишут результаты в
  `results/*.json`, а числа в этом README и в отчёте помечены невидимыми метками
  `<!--=ключ:формат-->`. `check_docs.py` сверяет их, `test_docs.py` делает то же
  в pytest и выборочно пересчитывает результаты.

## Структура репозитория

```
alakazam/
├── reentry/            пакет модели: atmosphere, trajectory, heating, ablation, emissivity
├── run_step*.py        шаги расчёта 1–5, explore_step1b.py, analysis_step5b.py
├── verify_step*.py     проверки (pytest или как скрипты)
├── check_docs.py       сверка чисел в документах с results/
├── test_docs.py        то же в pytest
├── results/            числа, которые пишут скрипты
├── figures/            графики
└── docs/REPORT.md      полный отчёт
```

## Главные допущения

Устойчивая ориентация фрагментов (предел быстрого кувыркания проверен отдельно);
три класса фрагментов, каждый термически считается алюминием; расплав удерживается
на фрагменте до кипения; вдув пара и теплота окисления в базу не входят; излучательная
способность голого жидкого Al — оценка по электросопротивлению; высоты разрушения —
входы, а не результат. Полный список — в [разделе 15 отчёта](docs/REPORT.md#15-полный-список-допущений).

## Как цитировать

Метаданные для цитирования — в [CITATION.cff](CITATION.cff) (GitHub показывает их
кнопкой «Cite this repository»).

## Лицензия

[MIT](LICENSE).
