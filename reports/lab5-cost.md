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

## 2. Actual, from billing

Read on **27 Oct 2026, 21:25 Bangkok**, from Cost Management → Cost analysis, subscription scope, daily, in USD. The screenshots are in `reports/lab5/cost-*.png`. Billing days are **UTC**, so 25 Oct on the bill runs from 07:00 on 25 Oct to 07:00 on 26 Oct, Bangkok time. Lab 5 ran from 25 Oct 16:42 UTC, when the estimate was committed, to 27 Oct 13:16 UTC, the teardown. **No capstone jobs ran from 25 to 27 Oct**, so every training charge in that window belongs to Lab 5.

| Billing day (UTC) | Container Registry | Virtual Machines | Storage | Azure Monitor | Virtual Network | Other | Total |
|---|---|---|---|---|---|---|---|
| 25 Oct | 0.17 | 0.04 | 0.01 | 0.02 | <0.01 | <0.01 | **0.25** |
| 26 Oct | 0.17 | 0.14 | 0.06 | 0.02 | 0.01 | <0.01 | **0.40** |
| 27 Oct (incomplete) | 0.06 | — | <0.01 | <0.01 | — | <0.01 | **0.07** |

On 26 Oct, grouped by resource, the workspace `itcs3556688020-ml` carries **$0.20**: the cluster's $0.14 of VM time, plus **$0.06** of storage and network billed with it. The other lines are standing charges that were there before Lab 5: the registry, and the capstone's alert rule `pm25-monitor-finding` (Azure Monitor, $0.02 a day). They also include the key vaults and the storage accounts.

**The `lab=5` tag cannot isolate this cost.** Azure ML bills a cluster's nodes to the **workspace** resource. The by-resource view has no row for `itcs355-lab5-cpu` at all, so the tag on the cluster never reaches the bill. That is the "untagged resources" failure mode, and here it comes from the platform rather than from forgetting to tag. Lab 5's cost was therefore attributed **by date and by resource**. That works only because nothing else used the cluster's workspace in that window.

| Lab 5 component | Billing basis | USD | THB |
|---|---|---|---|
| Training compute (VM) | VM lines, 25 + 26 Oct | 0.18 | 6.08 |
| Training: node disks and network | the workspace's $0.20 minus its $0.14 of VM, 26 Oct | 0.06 | 2.03 |
| Registry (shared) | $0.167/day × 1.86 days in Lab 5, on the same basis as the estimate | 0.31 | 10.47 |
| Serving (Azure staging app, AWS Lambda) | Container App: no line on the bill (scale-to-zero, inside the free grant). AWS Cost Explorer, 1 Sep – 27 Oct: **$0.00** | 0.00 | 0.00 |
| Storage, monitoring, schedule, LLM | storage accounts and Log Analytics no higher than before Lab 5; the pipeline and schedule have no charge of their own; Gemini free tier | 0.00 | 0.00 |
| **Lab 5 total** | | **0.55** | **≈ 18.6** |

27 Oct is incomplete on the bill. No Azure ML job ran that day (it was the AWS day), so the only charge still to arrive is the rest of the registry's flat fee, which the table already counts by rate.

**Term to date:** **$8.45 = 285.3 THB** on Azure, 1 Sep – 27 Oct, plus $0.00 on AWS. That is under the 800 THB limit, with 515 THB to spare.

## 3. The gap

**Estimate ≈ 55 THB, actual ≈ 18.6 THB: 36 THB (66%) under.** There are three causes, in order of size.

1. **Days, not runs, drove the estimate, and the lab took 1.9 days instead of 6.** The registry's flat fee was 34 of the estimated 55 THB, on the assumption that Part A would run until 31 Oct. It finished on 27 Oct, so the registry share is 10.5 THB. This also shows the estimate was framed wrongly. The registry is shared with the capstone and bills every day whether Lab 5 exists or not. The **marginal** cost of Lab 5, meaning what would not have been spent without it, is **8.1 THB**, against 21 THB estimated for the same components.
2. **Fewer, shorter runs.** The VM lines come to $0.18 = 1.14 node-hours. That covers 7 runs: run 1, the three failed register attempts, runs 2 and 3, and the scheduled firing. They took about 10 node-minutes each, against the 12 runs × 15 minutes assumed. Run 1 alone, on a cold cluster, was $0.04, which is 15 node-minutes.
3. **One charge the estimate missed, in the other direction.** A node is not just its VM rate. Its disk and network bill separately, $0.06 on 26 Oct, which **adds a third** on top of the VM time (0.06 / 0.14). The estimate priced training at the VM rate alone. It also allowed 5 THB for serving, monitoring and storage, which came to zero: the staging app scaled to zero, AWS stayed inside its free plan, and Log Analytics billed $0 on every day.

The **debugging cost** was smaller than feared. The three failed register attempts cost about 1 THB each, and the four Lambda calls that timed out at 60 s were free under the AWS plan. **The risk named in section 1, a node that does not scale down, did not happen.** The cluster stayed at 0–1 nodes, and no VM line on any day is larger than about one node-hour.

## 4. Breakdown by component

| Component | Estimate (THB) | Actual (THB) | Note |
|---|---|---|---|
| Training (pipeline steps on `itcs355-lab5-cpu`) | 16.0 | 8.1 | 6.1 VM time + 2.0 node disk and network |
| Storage (pipeline outputs, model artifacts) | 1.0 | 0.0 | storage accounts no higher than before Lab 5 |
| Serving (Azure staging app, AWS Lambda) | 2.0 | 0.0 | scale-to-zero and free plans; Lambda measured below |
| Pipeline orchestration and schedule | 0.0 | 0.0 | Azure ML charges only for the compute |
| Monitoring (Log Analytics) | 2.0 | 0.0 | Log Analytics $0 on every day |
| Registry (shared with the capstone) | 33.8 | 10.5 | flat fee, attributed by days |
| LLM step (Part B) | 0.0 | 0.0 | Gemini free tier, token counts in `reports/lab5-llm.md` |
| **Total** | **≈ 55** | **≈ 18.6** | |

## 5. Cost per 1,000 predictions at 5%, 25% and 80% utilisation

Two ways to serve the same model, priced at list rates rather than free grants, because a free grant measures the plan and not the design.

**Always-on Azure Container App** (one replica kept warm, 0.25 vCPU / 0.5 GiB, as in Lab 3):
- **Rate:** $0.000024 per vCPU-second and $0.000003 per GiB-second (active), which comes to $0.027 an hour = **0.912 THB/h**. Requests cost $0.40 per million after the 2-million monthly grant, which is 0.0135 THB per 1,000. These are the Azure Retail Prices API rates for East US as of 26 Oct 2026. Microsoft's pricing page did not render rates for Southeast Asia, so the regional figure is not confirmed.
- **Throughput:** 42.75 predictions/s measured in Lab 3, with utilisation applied to that.

**AWS Lambda** (1024 MB): cost per prediction comes from the billed duration AWS reported per invocation (`reports/lab5-lambda-rightsize.json`). The rates are $0.0000166667 per GB-second and $0.20 per million requests, AWS's published x86 rate (US East example). Sydney's rate is not listed on that page. **Utilisation does not enter**, because an idle function bills nothing.

| Utilisation | Container App: compute | + requests | **Container App total** | **Lambda, 1 per call** | **Lambda, 100 per call** |
|---|---|---|---|---|---|
| 5% | 0.1185 | 0.0135 | **0.132** | 0.0105 | 0.000135 |
| 25% | 0.0237 | 0.0135 | **0.037** | 0.0105 | 0.000135 |
| 80% | 0.0074 | 0.0135 | **0.021** | 0.0105 | 0.000135 |

All figures are THB per 1,000 predictions. Container App compute = 0.912 THB/h × (1,000 / (42.75 × utilisation)) s / 3,600.

**What it shows:**
- **At 5% utilisation the warm replica costs 12.6× what Lambda does per prediction.** Nearly all of that is paying for idle time. At 80% the gap shrinks to 2×, and on compute alone the replica (0.0074) is cheaper than Lambda (0.0105).
- **For single-reading calls on Lambda, the request fee is most of the bill.** $0.20 per million requests is 0.00675 of the 0.0105 THB, about 64%. Compute is only 6.6 ms a call. That is why batching, in section 6, is the lever that matters.
- **What staging actually ran was neither column.** The Container App scales to zero, so it bills only while a replica is up. At this course's traffic it stayed inside the monthly free grant, which is why serving is zero in section 4.

## 6. One optimisation applied: batch the fleet scoring

**Before:** scoring the fleet meant one `/predict` call per machine. **After:** `scripts/score_fleet.py` sends each machine's latest reading in batches of up to 100 to `/predict/batch` (`tests/test_portability.py` pins the batching: 240 rows go as 100 + 100 + 40). It was run against the Lambda on 27 Oct and scored **all 240 machines in 3 invocations instead of 240, in 3.2 s**, flagging 21 with a failure probability ≥ 0.40.

Every figure below is from AWS's own REPORT line for each invocation: 50 warm calls per configuration after one cold start, measured on 27 Oct (`scripts/lambda_rightsize.py`, `reports/lab5-lambda-rightsize.json`).

| | Before: 1 per call, 1024 MB | Tried and rejected: 1 per call, 512 MB | **Applied: 100 per call, 1024 MB** |
|---|---|---|---|
| Billed duration per call (mean) | 6.64 ms | 11.14 ms | 11.98 ms |
| Function duration p50 | 6.1 ms | 11.85 ms | 7.12 ms |
| Round trip at the client, p95 | 226 ms | 227 ms | 230 ms per call (2.3 ms per prediction) |
| Max memory used | 350 MB | 350 MB | 351 MB |
| GB-seconds per prediction | 0.00664 | 0.00557 | 0.000120 |
| **THB per 1,000 predictions** | **0.010488** | 0.009886 (−6%) | **0.000135 (−98.7%, 78× cheaper)** |

**Why it works.** Each call carries a fixed overhead: the request fee, the Web Adapter's HTTP hop, and FastAPI's routing and validation. The model's own work is tiny. A call with 100 rows bills 11.98 ms against 6.64 ms for one row, so 99 more predictions cost 5.3 ms. Both the request fee and the per-call overhead are divided by 100.

**Why not the smaller memory size**, the usual first move:
- The function never used more than 351 MB, so 512 MB fits.
- Lambda's CPU scales with memory, though, so the same work billed 1.7× longer and the GB-seconds fell only 16%. That is a 6% saving per prediction, and an earlier measurement of the same pair came out **1% more expensive**, so the saving is within the noise.
- Rejected: it halves the CPU for a saving that cannot be told apart from noise.

**What was given up.**
- **A reading waits for the next sweep** instead of being scored on arrival. That suits this system: the label is "fails within 7 days", and the decision it feeds, which machines to inspect, is made daily, not per reading.
- **One malformed row fails its whole batch of 100** (HTTP 422), where single calls would lose one machine. The scorer would need to retry such a batch row by row.
- **In baht the saving is tiny at this fleet's size:** 240 predictions a day cost 0.0025 THB before and 0.00003 THB after. At a million predictions a day the same change is 10.5 THB a day against 0.14.
