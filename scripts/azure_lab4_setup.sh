#!/usr/bin/env bash
# ITCS355 Lab 4 — one-time Azure + GitHub setup for OIDC CI/CD and drift monitoring.
#
#   az login                      # as you did for Labs 1–3
#   gh auth login                 # optional: lets this script set the GitHub secrets/vars
#   bash scripts/azure_lab4_setup.sh
#
# Safe to re-run: every step checks before it creates.
#
# What it creates, and why each is the minimum:
#   * a user-assigned managed identity `itcs355-github` — the identity GitHub Actions
#     becomes. No client secret exists for it, so there is nothing to leak or rotate.
#     (A managed identity rather than an app registration because university Entra
#     tenants often stop students registering apps; a managed identity only needs rights
#     on your own resource group.)
#   * federated credentials trusting exactly two GitHub subjects on this repository:
#       ref:refs/heads/main   — CI's image push and the scheduled drift check
#       environment:staging   — CD's deploy
#     A pull request, a fork, or any other branch cannot obtain a token.
#   * role assignments:
#       AcrPush                        on the registry         (push the SHA-tagged image)
#       Contributor                    on the resource group   (az containerapp update)
#       Monitoring Metrics Publisher   on the Container App    (drift scores as metrics)
#     Contributor is broader than strictly needed; updating a Container App also needs
#     join rights on its managed environment, and RG scope is the simplest scope that
#     covers both. The README notes the narrower alternative.
#   * a MONITORING_TOKEN secret on the Container App guarding /monitoring/window.
set -euo pipefail
# Git Bash on Windows rewrites arguments that start with "/" into Windows paths, which
# turns Azure resource IDs (/subscriptions/...) into C:/Program Files/Git/subscriptions/...
export MSYS_NO_PATHCONV=1
# ...and az.cmd/gh.exe output there can end in \r, which would corrupt every captured
# value. Strip it; pipefail still propagates the real exit code.
# A zip-installed Azure CLI on Windows is only `az.cmd`, which bash will not find as `az`.
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*)
    AZ_BIN="$(command -v az || command -v az.cmd || true)"
    [ -n "$AZ_BIN" ] || { echo "Azure CLI not found on PATH (looked for az and az.cmd)"; exit 1; }
    az() { "$AZ_BIN" "$@" | tr -d '\r'; }
    gh() { command gh "$@" | tr -d '\r'; } ;;
esac

REPO="${GITHUB_REPO:-mntpps/itcs355-lab}"
APP="${CONTAINER_APP:-itcs355-predict}"
IDENTITY="${IDENTITY_NAME:-itcs355-github}"
INSTANCE="${INSTANCE:-0.25:0.5Gi}"
MODEL_VERSION="${MODEL_VERSION:-2}"

cd "$(dirname "$0")/.."
[ -f cloud.env ] || { echo "cloud.env not found — run from the repo with your cloud.env in place"; exit 1; }
# Read KEY=value lines, ignoring comments and trailing "# ..." annotations.
# tr -d '\r': a cloud.env saved on Windows has CRLF endings, and a trailing \r would end
# up inside every az argument.
get() { tr -d '\r' < cloud.env | grep -E "^$1=" | head -1 | cut -d= -f2- | sed 's/[[:space:]]*#.*$//; s/[[:space:]]*$//'; }

RG="$(get PROJECT_ID)"; REGION="$(get REGION)"; REGISTRY="$(get CONTAINER_REGISTRY)"
SUB="$(get AZURE_SUBSCRIPTION_ID)"; MLFLOW_URI="$(get MLFLOW_TRACKING_URI)"
MODEL_NAME="$(get MODEL_REGISTRY_NAME)"; BLOB_URI="$(get BLOB_URI)"
ACR_NAME="${REGISTRY%%.*}"
[ -n "$SUB" ] || SUB="$(az account show --query id -o tsv)"
TENANT="$(az account show --query tenantId -o tsv)"
az account set --subscription "$SUB"

step() { printf '\n== %s\n' "$*"; }

step "Container App $APP in $RG"
if ! az containerapp show -n "$APP" -g "$RG" -o none 2>/dev/null; then
  # Lab 3 teardown deleted it. Recreate it with the Lab 3 image; CD replaces the image
  # with this commit's on the first push. An existing Container Apps environment in the
  # resource group is reused (none of its other apps are touched); only if there is none
  # is a new one created, tagged lab=4 so teardown removes it.
  WORKSPACE="$(get AZURE_ML_WORKSPACE)"
  [ -n "$WORKSPACE" ] || { echo "AZURE_ML_WORKSPACE missing from cloud.env"; exit 1; }
  ENV_NAME="$(az containerapp env list -g "$RG" --query "[0].name" -o tsv)"
  if [ -z "$ENV_NAME" ]; then
    ENV_NAME="itcs355-lab4-env"
    echo "creating environment $ENV_NAME (2-3 minutes)"
    az containerapp env create -n "$ENV_NAME" -g "$RG" -l "$REGION" --logs-destination none \
      --tags course=itcs355 student="$RG" lab=4 -o none
  fi
  echo "creating $APP in environment $ENV_NAME"
  az containerapp create -n "$APP" -g "$RG" --environment "$ENV_NAME" \
    --image "$REGISTRY@sha256:21d9f3cbded6cdf3023acc5f9057a89f9009d518f6227b6d7fc8293817cbe923" \
    --registry-server "${REGISTRY%%/*}" --registry-identity system --system-assigned \
    --ingress external --target-port 8080 --cpu 0.25 --memory 0.5Gi \
    --min-replicas 0 --max-replicas 1 \
    --env-vars "MODEL_REGISTRY_NAME=$MODEL_NAME" "MODEL_VERSION=$MODEL_VERSION" "MLFLOW_TRACKING_URI=$MLFLOW_URI" \
    --tags course=itcs355 student="$RG" lab=4 -o none
  # The app's own identity reads model v2 from the Azure ML registry at startup.
  APP_PRINCIPAL="$(az containerapp show -n "$APP" -g "$RG" --query identity.principalId -o tsv)"
  WS_ID="$(az ml workspace show -n "$WORKSPACE" -g "$RG" --query id -o tsv 2>/dev/null || \
    echo "/subscriptions/$SUB/resourceGroups/$RG/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE")"
  az role assignment create --assignee-object-id "$APP_PRINCIPAL" --assignee-principal-type ServicePrincipal \
    --role "AzureML Data Scientist" --scope "$WS_ID" -o none
  echo "granted AzureML Data Scientist on $WORKSPACE to the app's identity"
  echo "waiting 60 s for the role to propagate, then restarting so the model loads"
  sleep 60
  REV="$(az containerapp show -n "$APP" -g "$RG" --query properties.latestRevisionName -o tsv)"
  az containerapp revision restart -n "$APP" -g "$RG" --revision "$REV" -o none || true
fi
APP_ID="$(az containerapp show -n "$APP" -g "$RG" --query id -o tsv)"
FQDN="$(az containerapp show -n "$APP" -g "$RG" --query properties.configuration.ingress.fqdn -o tsv)"
echo "found https://$FQDN"

step "Staging shape: one replica, one window (see service/monitoring.py)"
# A fresh app has no secrets at all (null, not []), so test for the name, not a count.
TOKEN="$(az containerapp secret list -n "$APP" -g "$RG" --query "[?name=='monitoring-token'].name" -o tsv)"
if [ -z "$TOKEN" ]; then
  MONITORING_TOKEN="$(openssl rand -hex 24)"
  az containerapp secret set -n "$APP" -g "$RG" --secrets "monitoring-token=$MONITORING_TOKEN" -o none
  NEW_TOKEN=1
else
  MONITORING_TOKEN="$(az containerapp secret show -n "$APP" -g "$RG" --secret-name monitoring-token --query value -o tsv)"
  NEW_TOKEN=0
fi
az containerapp update -n "$APP" -g "$RG" --max-replicas 1 \
  --set-env-vars "MONITORING_TOKEN=secretref:monitoring-token" \
  --tags course=itcs355 student="$RG" lab=4 -o none
echo "max replicas 1, MONITORING_TOKEN wired from the app's secret store, tagged lab=4"

step "Managed identity $IDENTITY"
if ! az identity show -n "$IDENTITY" -g "$RG" -o none 2>/dev/null; then
  az identity create -n "$IDENTITY" -g "$RG" -l "$REGION" --tags course=itcs355 student="$RG" lab=4 -o none
fi
CLIENT_ID="$(az identity show -n "$IDENTITY" -g "$RG" --query clientId -o tsv)"
PRINCIPAL_ID="$(az identity show -n "$IDENTITY" -g "$RG" --query principalId -o tsv)"
echo "client id $CLIENT_ID"

step "Federated credentials (who may become this identity)"
# GitHub's OIDC subject now embeds the owner and repository IDs
# (repo:owner@123/name@456:...), so a deleted-and-recreated repository with the same
# name cannot inherit this trust. Register that form, plus the legacy name-only form in
# case the repository is set to use it. Azure matches the subject exactly.
IDS="${GITHUB_REPO_IDS:-$(gh api "repos/$REPO" --jq '"\(.owner.login)@\(.owner.id)/\(.name)@\(.id)"' 2>/dev/null || true)}"
PAIRS=("gh-main:repo:$REPO:ref:refs/heads/main" "gh-staging:repo:$REPO:environment:staging")
if [ -n "$IDS" ]; then
  PAIRS+=("gh-main-id:repo:$IDS:ref:refs/heads/main" "gh-staging-id:repo:$IDS:environment:staging")
else
  echo "  (gh unavailable: set GITHUB_REPO_IDS=owner@<id>/repo@<id> to register the ID-based subjects)"
fi
for pair in "${PAIRS[@]}"; do
  name="${pair%%:*}"; subject="${pair#*:}"
  if ! az identity federated-credential show -n "$name" --identity-name "$IDENTITY" -g "$RG" -o none 2>/dev/null; then
    az identity federated-credential create -n "$name" --identity-name "$IDENTITY" -g "$RG" \
      --issuer https://token.actions.githubusercontent.com --subject "$subject" \
      --audiences api://AzureADTokenExchange -o none
  fi
  echo "  $subject"
done

step "Role assignments"
ACR_ID="$(az acr show -n "$ACR_NAME" --query id -o tsv)"
RG_ID="$(az group show -n "$RG" --query id -o tsv)"
assign() {
  if [ "$(az role assignment list --assignee "$PRINCIPAL_ID" --role "$1" --scope "$2" --query 'length(@)' -o tsv)" = "0" ]; then
    az role assignment create --assignee-object-id "$PRINCIPAL_ID" --assignee-principal-type ServicePrincipal \
      --role "$1" --scope "$2" -o none
  fi
  echo "  $1  on  ${2##*/}"
}
assign AcrPush "$ACR_ID"
assign Contributor "$RG_ID"
assign "Monitoring Metrics Publisher" "$APP_ID"

step "GitHub: secrets, variables, and the staging environment for $REPO"
SECRETS=(
  "AZURE_CLIENT_ID=$CLIENT_ID"
  "AZURE_TENANT_ID=$TENANT"
  "AZURE_SUBSCRIPTION_ID=$SUB"
  "MONITORING_TOKEN=$MONITORING_TOKEN"
)
VARS=(
  "CLOUD_PROVIDER=azure" "PROJECT_ID=$RG" "REGION=$REGION" "BLOB_URI=$BLOB_URI"
  "CONTAINER_REGISTRY=$REGISTRY" "MLFLOW_TRACKING_URI=$MLFLOW_URI" "MODEL_REGISTRY_NAME=$MODEL_NAME"
  "IDENTITY_REF=$CLIENT_ID" "CONTAINER_APP=$APP" "INSTANCE=$INSTANCE" "MODEL_VERSION=$MODEL_VERSION"
  "STAGING_URL=https://$FQDN"
)
if command -v gh >/dev/null && gh auth status >/dev/null 2>&1; then
  for kv in "${SECRETS[@]}"; do printf '%s' "${kv#*=}" | gh secret set "${kv%%=*}" -R "$REPO"; done
  for kv in "${VARS[@]}"; do gh variable set "${kv%%=*}" -R "$REPO" --body "${kv#*=}"; done
  gh api -X PUT "repos/$REPO/environments/staging" >/dev/null
  echo "set ${#SECRETS[@]} secrets, ${#VARS[@]} variables, environment 'staging'"
  if ! gh secret list -R "$REPO" | grep -q ALERT_WEBHOOK_URL; then
    printf '\nPaste your Slack (or Discord) webhook URL for drift alerts, or press Enter to skip: '
    read -rs HOOK; echo
    [ -n "$HOOK" ] && printf '%s' "$HOOK" | gh secret set ALERT_WEBHOOK_URL -R "$REPO" && echo "ALERT_WEBHOOK_URL set"
  fi
else
  echo "gh not available — set these in GitHub > Settings > Secrets and variables > Actions:"
  echo "  Secrets:"; for kv in "${SECRETS[@]}"; do echo "    ${kv%%=*}"; done
  echo "    ALERT_WEBHOOK_URL   (your Slack webhook)"
  echo "  Variables:"; for kv in "${VARS[@]}"; do echo "    $kv"; done
  echo "  And create an environment named 'staging' (Settings > Environments)."
  echo "  Secret values: AZURE_CLIENT_ID=$CLIENT_ID AZURE_TENANT_ID=$TENANT AZURE_SUBSCRIPTION_ID=$SUB"
  [ "$NEW_TOKEN" = "1" ] && echo "  MONITORING_TOKEN=$MONITORING_TOKEN   (shown once; also stored as the app secret)"
fi

step "Local Prometheus target for the dashboard"
printf '[{"targets": ["%s:443"], "labels": {"env": "staging"}}]\n' "$FQDN" > monitoring/stack/targets/staging.local.json
echo "wrote monitoring/stack/targets/staging.local.json"

cat <<EOF

Done. Staging: https://$FQDN
Next:  git push origin main      -> CI, then CD deploys this commit to staging
       make monitor-up           -> dashboard at http://localhost:3000
EOF
