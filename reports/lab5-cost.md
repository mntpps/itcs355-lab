# Lab 5 — Cost report

Provider `azure` (southeastasia), with `aws` for the Task 4 swap and Gemini (free tier) for Part B.
Azure bills this subscription in USD. Every THB figure uses **33.76 THB/USD** (23 Oct 2026), the same rate as `reports/lab5-llm.md`.

## 1. Estimate, made before running

Committed on 2026-10-25, before Task 1 created anything. Sections 2–6 are filled in after teardown, and this section is not edited after that.

**Starting point.** Term spend, 1 Sep – 25 Oct 2026, from Cost Management → Cost analysis at billing-account scope, is **$7.78 = 262.65 THB**. This covers Labs 1–4 and the capstone, which share resource group `itcs355-6688020`. Azure's forecast for the same window is $7.87.

| Service (term to date) | USD | THB |
|---|---|---|
| Container Registry | 6.95 | 234.63 |
| Virtual Machines (Azure ML training) | 0.45 | 15.19 |
| Azure Monitor | 0.15 | 5.06 |
| Storage | 0.12 | 4.05 |
| Other (Virtual Network, credits) | 0.11 | 3.71 |

Almost the whole bill is the registry, a flat daily fee whether or not anything is pushed. All the training in the course so far cost 15 THB. So the estimate is driven mostly by **how many days Lab 5 takes**, and next by **how many pipeline runs**.

**Assumptions.**
- Lab 5 Part A runs from 26 to 31 Oct, which is 6 days including the 24-hour teardown recheck.
- Training runs on the existing cluster `itcs3556688020-cpu`: `Standard_DS2_v2`, `tier: dedicated`, 0–1 nodes, 120 s idle before scale-down (read from `az ml compute show` on 25 Oct). It uses dedicated rather than spot nodes because this subscription's low-priority quota is 0 vCPUs, and the increase request returned `ResourceNotAvailableForOffer` (Lab 2). The DS2_v2 Linux rate in southeastasia is $0.158/h = **5.33 THB/h**, now added to `src/costs.py`.
- The cluster was created on 12 Oct and already carries the tag `lab: 5`, so a billing filter on `lab=5` will also pick up its capstone use before today. Section 2 has to separate the two by date.
- Each pipeline run bills about **15 node-minutes**: about 4 to start a node from zero, about 9 for the steps, and 2 of idle time before scale-down.
- There are **12 runs**: 6 while getting the pipeline to work, 1 gate pass, 1 gate fail, 1 aborted on a failing data contract, 1 for the permission-removal test, and 2 fired by the schedule.

| Component | Basis | Estimate (THB) |
|---|---|---|
| Registry (ACR Basic) | $0.167/day × 6 days. The term bill implies $0.165/day. It is shared with the capstone and cannot be torn down with Lab 5. | 33.8 |
| Training (pipeline steps) | 12 runs × 0.25 h × 5.33 THB/h | 16.0 |
| Serving | Staging Container App scales to zero and stays inside the monthly free grant. The AWS Lambda, S3 and ECR usage is a few invocations and one image on a new account's free plan. | 2.0 |
| Monitoring | Log Analytics ingestion from pipeline and app logs, at the term's rate | 2.0 |
| Storage | Workspace blob storage: pipeline outputs and model artifacts in the MB range | 1.0 |
| Pipeline orchestration and schedule | Azure ML charges nothing beyond the compute | 0.0 |
| LLM step (Part B) | Gemini free tier, already measured | 0.0 |
| **Total for Lab 5** | | **≈ 55 THB** |

**Range: 40 to 85 THB.** The low end is 6 runs of 10 minutes. The high end is 25 runs of 20 minutes, which is what a hard-to-debug pipeline looks like. The registry's 34 THB is in every case, so this lab costs more in registry days than in training.

**Term projection:** 262.65 + 55 ≈ **318 THB**, well under the 800 THB limit, leaving about 482 THB of headroom. At the registry's 5.64 THB/day, it alone would take 85 more days to use that.

**What would break this estimate.** One mistake outweighs everything above: a compute node that does not scale down. A DS2_v2 left running costs **128 THB a day**, more than twice the whole estimate. The cluster is at `min_instances: 0` and `max_instances: 1`, which caps that at one node. It must stay that way, and Task 6 checks that no node or compute instance is left.
