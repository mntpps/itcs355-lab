# Lab 5 Task 2 — three identities, least privilege

`scripts/azure_lab5_setup.sh` creates all three as user-assigned managed identities. None of them has a password or key. CI gets its token through GitHub OIDC, and the others get theirs from the Azure resource they are attached to.

| Identity | Attached to | Role | Scope | Why it is needed |
|---|---|---|---|---|
| **itcs355-train** | Lab 5 compute cluster; every pipeline step except deploy | AzureML Data Scientist | workspace | log the training run's params, metrics and model artifact; read the production model; register the new version and set its tags |
| | | Storage Blob Data Contributor | workspace storage account | each step reads its inputs from, and writes its outputs to, the workspace datastore under the step's identity |
| | | AcrPull | registry | pull the training image the steps run |
| **itcs355-serve** | staging Container App | AcrPull | registry | pull the serving image when a revision starts |
| | | AzureML Data Scientist | workspace | load the model version named by `MODEL_VERSION` from the registry at startup |
| **itcs355-github** | GitHub Actions (OIDC: `main` and `environment:staging` only); the pipeline's deploy step | AcrPush | registry | push the serving and training images CI builds |
| | | AzureML Data Scientist | workspace | trigger the pipeline; on a failed smoke test, put the production tag back on the previous version |
| | | Contributor | the staging app only | create a new revision (CD's image deploy, the pipeline's model rollout) |
| | | ~~Storage Blob Data Reader~~ | workspace storage account | **removed**: believed needed to read the register step's output; the test below showed nothing uses it |
| | | ~~Container Apps environment join (custom)~~ | the Container Apps environment | **removed**: never actually in place, and nothing failed without it (below) |

**What changed from Lab 4.** The CI identity had Contributor on the whole resource group, so it could delete the workspace, the registry or the capstone. It now has three roles: push to the registry, use the workspace, and Contributor on the one staging app. Training and serving used to be either my own login or a system identity with broad workspace rights.

**Where the boundary leaks.** The deploy step runs as `itcs355-github`, so that identity is attached to the cluster. An identity attached to a cluster can be used by any job submitted to that cluster: anyone who can submit a job there can deploy to staging, whatever identity the job asks for. The real boundary is therefore "who may submit jobs to this cluster" (AzureML Data Scientist on the workspace), not the identity table above. Keeping the deployer off the cluster would mean the pipeline ends at register, with something outside Azure ML (CD, or an Event Grid trigger) doing the deploy.

**AzureML Data Scientist is broader than the job needs.** It can also submit jobs and edit any experiment. The training identity needs only metrics, artifacts and model versions, and the serving identity needs only to read one model. Azure has no built-in read-only model role, so the narrower version would be a custom role. That is the next tightening, and the test below is how to find out whether it holds.

## The removal tests

### 1. The environment join right: not needed, and the Lab 4 README was wrong

Lab 4's README claimed that updating a Container App "also needs join rights on its environment", which was the reason CD then had Contributor on the whole resource group. Lab 5 gave the CI identity a custom role holding exactly that right. On 27 Oct the plan was to remove it and watch CD fail. Instead, `az role assignment list --assignee <itcs355-github> --all` showed it **had never been in place**: four roles, none of them the join role, and `az role assignment list --scope <environment>` returned nothing at all. The setup script had printed it as assigned.

Without it, everything that changes staging had already worked:
- CD run for `abb4581`: `az containerapp update` with a new image, then the smoke test, green.
- CD run for `cc802ff`: the same, green.
- Pipeline run 3, `heroic_wolf_gkcg02y5wc`: `deploy_staging` PATCHed the app's template through Resource Manager (new revision, `MODEL_VERSION=3`), and the smoke test passed with `/ready` reporting v3.

**What this shows:** the join check applies when an app is *placed* into an environment (create, or a move), not when an app already in it gets a new revision. Contributor on the app alone is enough to deploy. The role, and the claim behind it, are gone from the setup script.

### 2. Storage Blob Data Reader on the deployer: not needed, and that is the leak

**Removed** from `itcs355-github` on 27 Oct, about 02:05. I believed the `deploy_staging` step needed it to read its inputs (the register step's output and the gate decision) from the workspace's storage under its own identity. **Test:** the schedule's test firing, `itcs355-retrain-weekly-20261026T193017Z` at 02:30. The deploy step mounts its inputs whatever the gate decided.

**What broke: nothing.** `deploy_staging` mounted both inputs, read the gate decision, and logged `condition gate_decision == pass is false: deploy_staging skipped`, green (`reports/lab5/schedule-fired-gate-fail.png`). The role stays removed.

**What this shows:** the step's own identity is not what reads the workspace's datastore. The most likely reason is that `workspaceblobstore`, like every workspace's default datastore unless it is created otherwise, authenticates with the storage account's key, which the workspace holds. `az ml datastore show` returns `"credentials": {}`, which is how the CLI shows a key-based datastore with its secret removed. That fits, but it is not conclusive. Either way, the test shows the consequence directly: inside an Azure ML job, an identity's storage roles do not decide what it can read from the workspace's datastore. **Any job submitted to the workspace can read every pipeline output**, whatever identity it runs as. The real data boundary is "who can submit jobs here" (AzureML Data Scientist), the same conclusion as the cluster-identity leak above. The training identity's Storage Blob Data Contributor is almost certainly unnecessary for the same reason. It is untested, so it stays.

**Closing that leak** would mean switching the datastore to identity-based access (`az ml datastore update` with no credentials). Every step's storage role would then really matter, and this test would fail instead of pass.
