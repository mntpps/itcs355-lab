# Lab 4 — evidence

Everything a grader needs to check without asking. The staging results come from the run
on 2026-10-25, with screenshots in `reports/lab4/`; everything else was measured from this
repository and can be re-run with the command shown.

## 1. Blocked bad commit (Task 3) — required

| | |
|---|---|
| Branch / PR | `lab4-bad-contract` → [PR #1](https://github.com/mntpps/itcs355-lab/pull/1), closed without merging |
| Change | `scripts/make_dataset.py` emits `temp_c` in Fahrenheit ("sensor vendor v2 feed") |
| Failing run | [CI run 40182494621](https://github.com/mntpps/itcs355-lab/actions/runs/40182494621), job [`test`](https://github.com/mntpps/itcs355-lab/actions/runs/40182494621/job/121136335092) failed after 51 s; `build` skipped |
| Stage that stopped it | **Data contract tests**. Secret scan, hash-checked install, lint, portability audit, dataset generation and unit tests all passed first. Model behaviour tests, build, integration and push never ran. |
| Test that caught it | `tests/test_data.py::test_features_within_plausible_ranges` |
| Error | `AssertionError: temp_c above plausible ceiling 140.0: 240.215` |
| Also failed | `test_feature_distributions_match_training_reference`: `temp_c: mean 175.24 is 9.42 SD from the training reference 79.67 (limit 0.25)`, and `test_serving_reference_profile_is_current` (the data no longer matches the drift reference in the image) |

Screenshots: `reports/lab4/blocked-pr.png` (the PR's checks: `test` failing, `build` skipped) and `reports/lab4/blocked-step.png` (the failing step's log).

Reproduce locally: `git switch lab4-bad-contract && make data && make test-contract`.

## 2. CI green, CD to staging (Task 2)

| | |
|---|---|
| CI run on main | [CI run 40201120128](https://github.com/mntpps/itcs355-lab/actions/runs/40201120128) for `70573ca`: every stage green, including the OIDC login and the image push |
| Image pushed | `itcs3556688020.azurecr.io/itcs355:70573ca840ff23fe24374a353d410df2c9847aa6` (digest in the run summary) |
| CD run | [CD run 40201618806](https://github.com/mntpps/itcs355-lab/actions/runs/40201618806): deploy, then the smoke test (3/3 payloads, `/ready` reporting the same SHA), then the drift job moved to the same build; rollback step skipped |
| Auth | OIDC federation to user-assigned managed identity `itcs355-github`; no stored key exists |

## 3. Dashboard (Task 4)

`docker compose -f monitoring/stack/docker-compose.yml up -d prometheus grafana`, then pick `staging` in the target dropdown. Screenshot: `reports/lab4/dashboard-drift.png` (last hour, 21:08–22:08 ICT). What it shows:

- **The shift.** The `temp_c` PSI line crosses the dashed 0.10 line at about 21:34, a minute after the injection. It peaks near 0.55 while the window holds half normal and half shifted readings (a two-humped shape PSI punishes most), then settles at 0.42. The rolling mean moves from the reference, 79.7 °C, to 85.5 °C. Every other feature stays near zero.
- **The window resetting on deploy.** The flat 0.42 line before 21:20 is the drifted window left over from the 17:12 injection. It vanishes at 21:20, when CD deployed `70573ca` and restarted the app, and that is the in-memory limitation the README describes.
- **The smoke test.** The 33% 4xx spike at 21:20–21:25 is CD's three-payload smoke test, one of which is the deliberate Fahrenheit reading the schema rejects.
- **Latency.** p95 stays around 21 ms, far inside the 200 ms target.

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
| Normal traffic fills the window | 21:30:02–21:33:10 (1,493 accepted, 7 deliberate 422s) |
| Manual check on the normal window | 21:33:18, execution `itcs355-drift-v66ysd2`: **Succeeded** (no drift, no false alarm) |
| Injection start (`reports/lab4-timeline.md`) | **21:33:28** (drifted replay finished 21:36:36) |
| Dashboard PSI crosses 0.10 | about **21:34**, about 1 min after injection (`reports/lab4/dashboard-drift.png`) |
| First scheduled check after injection | 21:45:00, `itcs355-drift-29882325`: detected the drift (exit 2), but the job's webhook secret did not post to the channel, so no alert was received |
| Scheduled check that alerted | **22:00:00**, `itcs355-drift-29882340`, after the webhook secret was corrected at 21:58 |
| Alert received in Slack | **22:00:21**: `temp_c` PSI 0.420 ≥ 0.10, KS 0.268, mean 79.67 → 85.47 °C, schema clean; screenshot `reports/lab4/alert.png` |
| **Detection time, injection → alert** | **26 min 53 s** end to end. The scheduled detector itself caught it after 11 min 32 s (21:45); the remaining 15 min were the misconfigured alert channel. |

Expected bound: up to 15 minutes from the job's cron, plus about a minute of start-up. The first attempt used GitHub's schedule instead. It never fired in the hour after an injection at 17:12:59 ICT on 2026-10-25, which is why the check now runs as an Azure Container Apps job. Job executions and their start times: `az containerapp job execution list -n itcs355-drift -g itcs355-6688020 -o table`. In the local rehearsal (15 readings/s, same code), the dashboard PSI went from 0.06 to 0.22 between 15 and 30 s after injection, and the detector's PSI for the full drifted window was 0.34. The service's rolling PSI and the detector's PSI agreed to four decimal places, because they share `src/drift_stats.py`.

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

Run on 2026-10-25 at about 22:12 ICT, after the evidence above was captured:

```
> get_adapter(cfg).teardown(cfg.tags(4))
['itcs355-predict', 'itcs355-drift', 'itcs355-github']

> python scripts/teardown_verify.py --lab 4
searching for resources tagged {'course': 'itcs355', 'student': 'itcs355-6688020', 'lab': '4'}
PASS  nothing found under these tags

> az containerapp job list -g itcs355-6688020 -o table
itcs3556688020-monitor / -refresh / -retrain     (the capstone's jobs, not Lab 4's)

> docker compose -f monitoring/stack/docker-compose.yml down
grafana, prometheus, network removed
```

The drift job was deleted along with the staging app, so no schedule outlives the endpoint it calls. The GitHub `Drift check` workflow is manual-only and cannot run on its own. With the CI/CD identity gone, a push to `main` would now fail at the image push; `scripts/azure_lab4_setup.sh` recreates the identity, app and job if a later lab needs them.
