# Post-mortem — temp_c reads 6 °C hot across the whole fleet

Injected on purpose in Task 6 (`make replay-drift TARGET_URL=<staging>`: `temp_c` shifted +6 °C). Timeline, screenshots and the evidence behind the numbers are in [`lab4-evidence.md`](lab4-evidence.md); cost figures come from [`lab4-threshold.json`](lab4-threshold.json) and the model in this repo.

**What fired:** The scheduled drift check posted to Slack at ‹ALERT TIME ICT›: `temp_c` PSI 0.34 ≥ 0.10 (KS 0.24, window mean 79.7 → 85.2 °C), with schema, nulls and ranges clean. That was ‹N› min after the injection at ‹INJECTION TIME›. The dashboard's rolling PSI crossed 0.10 within about 30 s.

**True cause:** A calibration offset upstream, not the world changing. Every machine moved by exactly +6 °C at the same instant, while vibration, pressure and load did not move with it, and real overheating would. It is a broken producer whose output still passes the schema (85 °C is plausible), so it looks like data drift.

**Retrain, roll back, or no action — and why:** Neither retrain nor roll back the model, because the model is fine and its inputs are wrong. Fix the sensor feed, then re-score the affected window. Retraining on hot readings teaches the model that 85 °C is normal. Once the sensors are fixed, it would flag 16.7% of failing machines instead of 19.3%, which is permanent damage from a temporary fault. A gradual shift on *some* machines, with failures rising, would be real heating, and that is the case for retraining.

**What this would have cost if unnoticed for a week:** Flagged readings rise from 3.67% to 4.29%. At 240 machines scored daily (1,680 readings a week) and about THB 2,000 per needless inspection, that is about 10 extra inspections, roughly **THB 21,000**. The reflexive retrain costs far more: about 5 fewer warnings a week on machines that do fail is roughly one unplanned breakdown (about THB 150,000) a week, every week the corrupted model serves.

**How to prevent or detect it faster:** Alert on the dashboard series directly with a Prometheus rule, `feature_window_psi / feature_psi_threshold > 1 for 2m`, scraped every 5 s. That cuts detection from "next 15-minute run plus GitHub's delay" to about 3 minutes. Also require a `calibration_id` in the sensor payload, so a recalibration announces itself.
