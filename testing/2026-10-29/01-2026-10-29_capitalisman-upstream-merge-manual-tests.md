# Capitalisman upstream merge — manual tests (CMAN-D1-021)

Mark each box `[x]` when the step passed, or paste what you saw under it. Send the file
back as `01-2026-10-29_capitalisman-upstream-merge-manual-tests_reviewed.md`. Ticket CMAN-D1-021 closes only on your marks.

**Verified against:** c69e8e6
**Covers:** backtesting/metrics.py, ui/page_backtest.py, ui/page_predict.py, ui/components.py, ui/page_simulator.py, ui/page_screener.py, indicators/_utils.py, signals/combiner.py, data/fetcher.py, run-timesfm.sh, requirements.txt, CLAUDE.md.ref
**Branch under test:** develop — this check is only valid on a checkout that carries the
code (a different branch runs different code; verify after the merge, not before).

What changed: the friend's `main` (his correctness review, research suite, TimesFM 3) is merged
into `develop`. Where both sides fixed the same thing his version won; ours was kept for the
cross-asset `auto_adjust=False` fix, the fetcher retry and screener batch fetch, the page
navigation fix, staleness warnings and `run-timesfm.sh`. Nothing was pushed.

## Prerequisites

- [ ] A terminal in `/home/simplex/projects/Capitalisman` with the venv active (prompt starts with `(venv)`):
      ```
      cd /home/simplex/projects/Capitalisman
      source venv/bin/activate
      ```
      Inside a Claude session the venv is not active: use `venv/bin/python` instead of `python`.
- [ ] `git branch --show-current` prints `develop`.
- [ ] `git log --oneline -1` prints a line starting `c69e8e6` or a later commit. (No `git pull`:
      the merge is local only and not pushed.)

---

## A. The merge itself

- [ ] **A1. All four validation checks pass on the merged code.**
      ```
      cd /home/simplex/projects/Capitalisman && source venv/bin/activate && python -m pip check && python -m ruff check . && python -m pytest && git diff --check && echo ALL-GREEN
      ```
      Pass: `No broken requirements found.`, `All checks passed!`, a line `452 passed in` (seconds vary), then `ALL-GREEN`.
      Fail: any `FAILED`/`error` line, or no `ALL-GREEN`. Paste the last lines under this box.

- [ ] **A2. `develop` now contains every commit of the friend's `main`.**
      ```
      cd /home/simplex/projects/Capitalisman && git rev-list --count develop..main
      ```
      Pass: `0`. (It was `9` before the merge.)
      Fail: any other number.

- [ ] **A3. The confidence scorer is exactly the friend's version.**
      ```
      cd /home/simplex/projects/Capitalisman && git diff --quiet main develop -- signals/combiner.py && echo SAME-AS-MAIN || echo DIFFERS
      ```
      Pass: `SAME-AS-MAIN`.
      Fail: `DIFFERS`.

- [ ] **A4. Our cross-asset lookahead fix survived the merge.**
      ```
      cd /home/simplex/projects/Capitalisman && grep -n "auto_adjust=False" indicators/_utils.py
      ```
      Pass: one line, `24:            df = yf.Ticker(ticker).history(period=period, auto_adjust=False)`. <!-- src: indicators/_utils.py fetch_reference_close -->
      Fail: no output.

- [ ] **A5. The brain's ticket DB and agent scratch papers stay out of git.**
      ```
      cd /home/simplex/projects/Capitalisman && git check-ignore database/tickets.db .work/CMAN-D1-021
      ```
      Pass: both paths printed back, `database/tickets.db` and `.work/CMAN-D1-021`.
      Fail: either one missing from the output.

---

## B. The app

- [ ] **B1. The app starts.** Leave this terminal running for B2 to B4.
      ```
      cd /home/simplex/projects/Capitalisman && source venv/bin/activate && python -m streamlit run app.py --server.address 127.0.0.1
      ```
      Pass: Streamlit prints a URL with port `8501`, and opening `http://127.0.0.1:8501` shows the sidebar
      with Predict, Backtest, Simulator, Explore, Search, Compare, Screener. <!-- src: app.py page radio -->
      Fail: a Python traceback in the terminal, or a red error box on the page.

- [ ] **B2. Backtest page uses the friend's daily-equity metrics.** Sidebar → Backtest, ticker `AAPL`,
      click **Run Backtest**.
      Pass: under the results a caption containing `Signals use completed closes and enter at the next open.`,
      and hovering the **Sharpe Ratio** help icon shows `Daily portfolio returns including cash days; zero risk-free rate, 252 stock / 365 crypto days per year`. <!-- src: ui/page_backtest.py render metrics -->
      Fail: a traceback, a red error box, or the old help text mentioning a risk-free rate percentage and a minimum trade count.

- [ ] **B3. Simulator steps, pauses and resets.** Sidebar → Simulator, ticker `AAPL`, click **Create replay**,
      then **Step one bar** twice, then **Reset replay**.
      Pass: each step moves the replay forward one bar; Reset returns to the setup form with the
      **Create replay** button. <!-- src: ui/page_simulator.py _render_controls, render -->
      Fail: a traceback or a button that does nothing.

- [ ] **B4. Screener scans with one batched download.** Sidebar → Screener, pick any preset watchlist,
      click **Scan Watchlist**.
      Pass: a results table with a signal for each ticker, no traceback. <!-- src: ui/page_screener.py render -->
      Expected, do not report: in the CSV export the **Name** column repeats the ticker. Our earlier
      rate-limit fix (CMAN-D1-004) dropped the per-ticker company-name lookup on purpose, because one
      extra request per ticker was what got the scan throttled.
      Fail: a traceback, or every ticker listed as failed.
      Stop the app afterwards with Ctrl+C in the B1 terminal.

- [ ] **B5. The TimesFM launcher still starts with the TimesFM environment.**
      ```
      cd /home/simplex/projects/Capitalisman && timeout 25 ./run-timesfm.sh --server.headless true; echo "stopped by timeout (expected)"
      ```
      Pass: the first two lines are `Launching with TimesFM (/home/simplex/.venvs/capitalisman-timesfm)` and
      `  context=8192 batch=128 device=cuda`, then Streamlit's URL lines, then the timeout line. <!-- src: run-timesfm.sh -->
      Fail: `TimesFM environment not found at …` (the launcher fell back to `venv/`), or a traceback.

---

## C. The project instructions

- [ ] **C1. The ref file carries the friend's project facts itself, his AGENTS.md is untouched, and his CLAUDE.md is gone.**
      ```
      cd /home/simplex/projects/Capitalisman && grep -c "not instructions for simplex_mind sessions\|Research facts (copied from upstream's AGENTS.md)" CLAUDE.md.ref; git diff --quiet upstream/main develop -- AGENTS.md && echo AGENTS-UNTOUCHED; test -e CLAUDE.md && echo CLAUDE-MD-PRESENT || echo CLAUDE-MD-GONE
      ```
      Pass: `2` (both new headings), then `AGENTS-UNTOUCHED`, then `CLAUDE-MD-GONE`. <!-- src: CLAUDE.md.ref Upstream agent files, Research facts -->
      Fail: a number below `2`, no `AGENTS-UNTOUCHED` line, or `CLAUDE-MD-PRESENT`.

- [ ] **C2. Read the new ref file once and say whether it describes the project the way you use it.**
      ```
      cd /home/simplex/projects/Capitalisman && less CLAUDE.md.ref
      ```
      Pass: the envs (`venv/`, `~/.venvs/capitalisman-timesfm`), the launch commands, the research facts and the guardrails match
      how you run Capitalisman, and nothing in it would override how simplex_mind works.
      Fail: anything wrong or missing; write it under this box.

---

## Known limits

- Accepted limit (CMAN-D1-021 F5): the friend's own CI job is red on his Python 3.11 lock.
  `test_timesfm_context_ui` fails with the lock's streamlit 1.32.2, and it fails the same way on his `main`.
  Our local venv (streamlit 1.59.2) passes every test. A push to origin would show the same red CI until upstream bumps the lock.
- The TimesFM 3 context selector (**TimesFM Historical Context** on Predict) only appears with the
  experimental TimesFM 3 backend, which needs its own environment from `requirements-timesfm3.txt`.
  That environment is not installed here, so this checklist does not cover it.
- The screener's batch fetch does not retry when Yahoo rate-limits it: `yf.download` swallows the error.
  This is older than the merge and is ticketed as CMAN-D1-022.
- Accepted limit (CMAN-D1-021 F11): `run-timesfm.sh` still listens on every network interface, as it did before the merge;
  the friend's AGENTS.md prefers 127.0.0.1. Adding `--server.address 127.0.0.1` to the launcher is a one-line change if you want it local-only.
- Nothing was pushed to origin; GitHub CI did not run.

## Dev command appendix

| Command | What it does |
|---|---|
| `source venv/bin/activate` | Activates the project venv for this terminal window |
| `python -m pip check` / `python -m ruff check .` / `python -m pytest` / `git diff --check` | The four validation checks from the friend's AGENTS.md; offline, no spend |
| `git rev-list --count develop..main` | Counts friend's commits not yet on develop |
| `git diff --quiet main develop -- <file>` | Exit 0 when the file is identical on both branches |
| `git check-ignore <paths>` | Prints the paths git ignores |
| `python -m streamlit run app.py --server.address 127.0.0.1` | Starts the app on this machine only; fetches public Yahoo data when you run pages |
| `timeout 25 ./run-timesfm.sh --server.headless true` | Starts the app in the TimesFM environment for 25 seconds |

Nothing here writes anything unless a step says so.
