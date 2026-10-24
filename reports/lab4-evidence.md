# Lab 4 — evidence

Everything a grader needs to check without asking. Items marked **‹fill›** come from the
staging run and are filled in once it has happened; everything else was measured from
this repository and can be re-run with the command shown.

## 1. Blocked bad commit (Task 3) — required

| | |
|---|---|
| Branch / PR | `lab4-bad-contract` → ‹fill: PR URL›, closed without merging |
| Change | `scripts/make_dataset.py` emits `temp_c` in Fahrenheit ("sensor vendor v2 feed") |
| Failing run | ‹fill: Actions run URL› |
| Stage that stopped it | **Data contract tests**, after secret scan, lint, portability audit and 27 unit tests passed. Build, integration and push never ran. |
| Test that caught it | `tests/test_data.py::test_features_within_plausible_ranges` |
| Error | `AssertionError: temp_c above plausible ceiling 140.0: 240.215` |
| Also failed | `test_feature_distributions_match_training_reference`: `temp_c: mean 175.24 is 9.42 SD from the training reference 79.67 (limit 0.25)`, and `test_serving_reference_profile_is_current` (the data no longer matches the drift reference in the image) |

Screenshot of the failed run: ‹fill: `reports/lab4/blocked-pr.png`›

Reproduce locally: `git switch lab4-bad-contract && make data && make test-contract`.

## 2. CI green, CD to staging (Task 2)

| | |
|---|---|
| CI run on main | ‹fill: URL› — every stage green |
| Image pushed | ‹fill: `itcs3556688020.azurecr.io/itcs355:<sha>` and digest from the run summary› |
| CD run | ‹fill: URL› — deploy, then the smoke test: 3/3 payloads, `/ready` reporting the same SHA |
| Auth | OIDC federation to user-assigned managed identity `itcs355-github`; no stored key exists |

## 3. Dashboard (Task 4)

`make monitor-up`, then pick `staging` in the target dropdown. Screenshots:
‹fill: `reports/lab4/dashboard-baseline.png`› (before injection) and
‹fill: `reports/lab4/dashboard-drift.png`› (after: the temp_c PSI line crosses the dashed 0.10, and the rolling mean leaves the reference).

All ten panel queries were checked against a real Prometheus 3.5 scraping the service: every one returns data.

## 4. Injected drift (Task 6)

Procedure, against staging:

```bash
make replay TARGET_URL=https://<staging-fqdn>          # fill the window with normal traffic
# wait for one scheduled drift run to report "ok"
make replay-drift TARGET_URL=https://<staging-fqdn>    # stamps reports/lab4-timeline.md
# wait for the Slack alert
```

| Event | Time (ICT) |
|---|---|
| Injection start (`reports/lab4-timeline.md`) | ‹fill› |
| Dashboard PSI crosses 0.10 | ‹fill› |
| Scheduled drift run that fired | ‹fill: run URL and start time› |
| Alert received in Slack | ‹fill› — screenshot ‹`reports/lab4/alert.png`› |
| **Detection time, injection → alert** | ‹fill› |

Expected bound: up to 15 minutes from the schedule, plus GitHub's own delay on scheduled runs. In the local rehearsal (15 readings/s, same code), the dashboard PSI went from 0.06 to 0.22 between 15 and 30 s after injection, and the detector's PSI for the full drifted window was 0.34. The service's rolling PSI and the detector's PSI agreed to four decimal places, because they share `src/drift_stats.py`.

## 5. Which statistic catches which fault

`make drift-modes` (whole file against the training reference):

| Mode | temp_c mean | PSI | KS | Verdict |
|---|---|---|---|---|
| none (control) | 79.58 | 0.002 | 0.010 | stable |
| `shift +6` | 85.58 | 0.381 | 0.250 | **DRIFT** |
| `scale ×1.5` | 79.58 (unchanged) | 0.183 | 0.111 | **DRIFT**, caught by PSI from shape alone |
| `mix ×5` | 80.17 | 0.014 | 0.040 | stable — below threshold |

On 500-reading windows (`make threshold-study`), the chosen rule catches `shift +3` 78% of the time per check, `shift +4` 99%, `scale ×1.5` 98% and `mix ×5` 53%. Using 0.25 on every feature instead catches `shift +4` 5% of the time and false-pages about 10 times a month.

## 6. Teardown

`make teardown && make cost-report` output: ‹fill›. Confirm in the Actions tab that **Drift check** shows *disabled*.
