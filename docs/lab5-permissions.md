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
| | | Storage Blob Data Reader | workspace storage account | read the register step's output (which version to deploy) |
| | | Contributor | the staging app only | create a new revision (CD's image deploy, the pipeline's model rollout) |
| | | ITCS355 Container Apps environment join (custom) | the Container Apps environment | `managedEnvironments/join/action` and `read`, the only rights a revision needs on the shared environment |

**What changed from Lab 4.** The CI identity had Contributor on the whole resource group, so it could delete the workspace, the registry or the capstone. It now has Contributor on one app plus a two-action custom role on the environment. Training and serving used to be either my own login or a system identity with broad workspace rights.

**Where the boundary leaks.** The deploy step runs as `itcs355-github`, so that identity is attached to the cluster. An identity attached to a cluster can be used by any job submitted to that cluster: anyone who can submit a job there can deploy to staging, whatever identity the job asks for. The real boundary is therefore "who may submit jobs to this cluster" (AzureML Data Scientist on the workspace), not the identity table above. Keeping the deployer off the cluster would mean the pipeline ends at register, with something outside Azure ML (CD, or an Event Grid trigger) doing the deploy.

**AzureML Data Scientist is broader than the job needs.** It can also submit jobs and edit any experiment. The training identity needs only metrics, artifacts and model versions, and the serving identity needs only to read one model. Azure has no built-in read-only model role, so the narrower version would be a custom role. That is the next tightening, and the test below is how to find out whether it holds.

## The removal test

_To be filled after the test is run: which permission was removed, the exact error, what broke and what did not, and whether the permission was restored or left out._
