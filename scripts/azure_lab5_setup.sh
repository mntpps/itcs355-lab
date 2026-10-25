#!/usr/bin/env bash
# ITCS355 Lab 5 — identities, roles and the staging app the training pipeline needs.
#
#   az login
#   bash scripts/azure_lab5_setup.sh
#
# Safe to re-run: every step checks before it creates. Everything it creates is tagged
# lab=5, so `make teardown LAB=5` removes it. It never touches the capstone's resources:
# the existing Container Apps environment is reused, not modified, and the pipeline gets
# its own compute cluster rather than changing the identities of the capstone's.
#
# Three identities, one job each (Task 2). docs/lab5-permissions.md has the table with the
# reason for every role.
#
#   itcs355-train   runs ingest, validate, train, evaluate and register on the cluster
#       AzureML Data Scientist          workspace          log metrics and artifacts, register models
#       Storage Blob Data Contributor   workspace storage  read and write step inputs and outputs
#       AcrPull                         registry           pull the training image
#
#   itcs355-serve   the staging Container App's identity
#       AcrPull                         registry           pull the serving image
#       AzureML Data Scientist          workspace          read the model version from the registry
#
#   itcs355-github  CI (GitHub OIDC, no secret) and the pipeline's deploy_staging step
#       AcrPush                         registry           push the serving and training images
#       AzureML Data Scientist          workspace          trigger the pipeline; re-tag on rollback
#       Storage Blob Data Reader        workspace storage  read the register step's output
#       Contributor                     staging app ONLY   create a new revision (deploy)
#       ITCS355 environment join        Container Apps env the join action a revision needs
#
# The Lab 5 cluster carries itcs355-train and itcs355-github; each pipeline step names which
# one it runs as (pipeline/pipeline.yaml `identities`).
set -euo pipefail
export MSYS_NO_PATHCONV=1
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*)
    AZ_BIN="$(command -v az || command -v az.cmd || true)"
    [ -n "$AZ_BIN" ] || { echo "Azure CLI not found on PATH (looked for az and az.cmd)"; exit 1; }
    az() { "$AZ_BIN" "$@" | tr -d '\r'; } ;;
esac

REPO="${GITHUB_REPO:-mntpps/itcs355-lab}"
APP="${CONTAINER_APP:-itcs355-predict}"
CLUSTER="${LAB5_CLUSTER:-itcs355-lab5-cpu}"
# The last serving image CD pushed (Lab 4, commit 70573ca). Any later SHA works too.
SERVE_IMAGE_TAG="${SERVE_IMAGE_TAG:-70573ca840ff23fe24374a353d410df2c9847aa6}"

cd "$(dirname "$0")/.."
[ -f cloud.env ] || { echo "cloud.env not found — run from the repo with your cloud.env in place"; exit 1; }
get() { tr -d '\r' < cloud.env | grep -E "^$1=" | head -1 | cut -d= -f2- | sed 's/[[:space:]]*#.*$//; s/[[:space:]]*$//'; }

RG="$(get PROJECT_ID)"; REGION="$(get REGION)"; REGISTRY="$(get CONTAINER_REGISTRY)"
SUB="$(get AZURE_SUBSCRIPTION_ID)"; MLFLOW_URI="$(get MLFLOW_TRACKING_URI)"
MODEL_NAME="$(get MODEL_REGISTRY_NAME)"; WS="$(get AZURE_ML_WORKSPACE)"
ACR_NAME="${REGISTRY%%.*}"
[ -n "$WS" ] || { echo "AZURE_ML_WORKSPACE missing from cloud.env"; exit 1; }
az account set --subscription "$SUB"
TAGS=(course=itcs355 "student=$RG" lab=5)

step() { printf '\n== %s\n' "$*"; }

az extension show -n ml -o none 2>/dev/null || az extension add -n ml -o none

step "Resources these roles are scoped to"
WS_ID="$(az ml workspace show -n "$WS" -g "$RG" --query id -o tsv)"
STORAGE_ID="$(az ml workspace show -n "$WS" -g "$RG" --query storage_account -o tsv)"
ACR_ID="$(az acr show -n "$ACR_NAME" --query id -o tsv)"
ENV_ID="$(az containerapp env list -g "$RG" --query "[0].id" -o tsv)"
[ -n "$ENV_ID" ] || { echo "no Container Apps environment in $RG"; exit 1; }
echo "workspace  ${WS_ID##*/}"
echo "storage    ${STORAGE_ID##*/}"
echo "registry   ${ACR_ID##*/}"
echo "app env    ${ENV_ID##*/}   (reused, not modified)"

step "Identities"
declare -A CLIENT PRINCIPAL RID
for name in itcs355-train itcs355-serve itcs355-github; do
  if ! az identity show -n "$name" -g "$RG" -o none 2>/dev/null; then
    az identity create -n "$name" -g "$RG" -l "$REGION" --tags "${TAGS[@]}" -o none
  fi
  CLIENT[$name]="$(az identity show -n "$name" -g "$RG" --query clientId -o tsv)"
  PRINCIPAL[$name]="$(az identity show -n "$name" -g "$RG" --query principalId -o tsv)"
  RID[$name]="$(az identity show -n "$name" -g "$RG" --query id -o tsv)"
  echo "  $name  client ${CLIENT[$name]}"
done

step "Custom role: join the Container Apps environment, and nothing else"
JOIN_ROLE="ITCS355 Container Apps environment join"
if [ -z "$(az role definition list --name "$JOIN_ROLE" --query "[].id" -o tsv)" ]; then
  mkdir -p pipeline/build
  cat > pipeline/build/join-role.json <<EOF
{
  "Name": "$JOIN_ROLE",
  "Description": "Create Container App revisions in an existing environment. ITCS355 Lab 5.",
  "Actions": ["Microsoft.App/managedEnvironments/join/action", "Microsoft.App/managedEnvironments/read"],
  "AssignableScopes": ["/subscriptions/$SUB/resourceGroups/$RG"]
}
EOF
  az role definition create --role-definition @pipeline/build/join-role.json -o none
  echo "created; waiting 30 s for it to replicate"
  sleep 30
fi
echo "  $JOIN_ROLE"

step "Staging app $APP"
if ! az containerapp show -n "$APP" -g "$RG" -o none 2>/dev/null; then
  # Before the app exists the serve identity needs AcrPull, or the first revision cannot
  # pull its image.
  if [ -z "$(az role assignment list --assignee "${PRINCIPAL[itcs355-serve]}" --role AcrPull --scope "$ACR_ID" --query "[].id" -o tsv)" ]; then
    az role assignment create --assignee-object-id "${PRINCIPAL[itcs355-serve]}" \
      --assignee-principal-type ServicePrincipal --role AcrPull --scope "$ACR_ID" -o none
    echo "granted AcrPull to itcs355-serve; waiting 60 s for it to propagate"
    sleep 60
  fi
  az containerapp create -n "$APP" -g "$RG" --environment "$ENV_ID" \
    --image "$REGISTRY:$SERVE_IMAGE_TAG" \
    --registry-server "${REGISTRY%%/*}" --registry-identity "${RID[itcs355-serve]}" \
    --user-assigned "${RID[itcs355-serve]}" \
    --ingress external --target-port 8080 --cpu 0.25 --memory 0.5Gi \
    --min-replicas 0 --max-replicas 1 \
    --env-vars "MODEL_REGISTRY_NAME=$MODEL_NAME" "MODEL_VERSION=2" "MLFLOW_TRACKING_URI=$MLFLOW_URI" \
      "AZURE_CLIENT_ID=${CLIENT[itcs355-serve]}" "BUILD_SHA=$SERVE_IMAGE_TAG" \
    --tags "${TAGS[@]}" -o none
  echo "created $APP serving model v2 as itcs355-serve"
fi
APP_ID="$(az containerapp show -n "$APP" -g "$RG" --query id -o tsv)"
FQDN="$(az containerapp show -n "$APP" -g "$RG" --query properties.configuration.ingress.fqdn -o tsv)"
echo "  https://$FQDN"

step "Role assignments"
assign() {  # identity role scope
  local p="${PRINCIPAL[$1]}" existing
  existing="$(az role assignment list --assignee "$p" --role "$2" --scope "$3" --query "[].id" -o tsv)"
  if [ -z "$existing" ]; then
    az role assignment create --assignee-object-id "$p" --assignee-principal-type ServicePrincipal \
      --role "$2" --scope "$3" -o none
  fi
  printf '  %-15s %-32s on %s\n' "$1" "$2" "${3##*/}"
}
assign itcs355-train  "AzureML Data Scientist"        "$WS_ID"
assign itcs355-train  "Storage Blob Data Contributor" "$STORAGE_ID"
assign itcs355-train  AcrPull                         "$ACR_ID"
assign itcs355-serve  AcrPull                         "$ACR_ID"
assign itcs355-serve  "AzureML Data Scientist"        "$WS_ID"
assign itcs355-github AcrPush                         "$ACR_ID"
assign itcs355-github "AzureML Data Scientist"        "$WS_ID"
assign itcs355-github "Storage Blob Data Reader"      "$STORAGE_ID"
assign itcs355-github Contributor                     "$APP_ID"
assign itcs355-github "$JOIN_ROLE"                    "$ENV_ID"

step "Federated credentials: GitHub Actions may become itcs355-github"
IDS="${GITHUB_REPO_IDS:-mntpps@194188007/itcs355-lab@1368344234}"
for pair in "gh-main:repo:$REPO:ref:refs/heads/main" "gh-staging:repo:$REPO:environment:staging" \
            "gh-main-id:repo:$IDS:ref:refs/heads/main" "gh-staging-id:repo:$IDS:environment:staging"; do
  name="${pair%%:*}"; subject="${pair#*:}"
  if ! az identity federated-credential show -n "$name" --identity-name itcs355-github -g "$RG" -o none 2>/dev/null; then
    az identity federated-credential create -n "$name" --identity-name itcs355-github -g "$RG" \
      --issuer https://token.actions.githubusercontent.com --subject "$subject" \
      --audiences api://AzureADTokenExchange -o none
  fi
  echo "  $subject"
done

step "Compute cluster $CLUSTER (Lab 5's own)"
# Same size and limits as the capstone's cluster (Standard_DS2_v2, 0-1 nodes, 120 s idle),
# but separate: an Azure ML cluster takes either a system identity or user-assigned ones,
# and switching the capstone's would change what its jobs run as. Dedicated tier because
# this subscription's low-priority quota is 0 vCPUs.
IDS_CSV="${RID[itcs355-train]},${RID[itcs355-github]}"
if ! az ml compute show -n "$CLUSTER" -g "$RG" -w "$WS" -o none 2>/dev/null; then
  az ml compute create -n "$CLUSTER" -g "$RG" -w "$WS" --type amlcompute \
    --size Standard_DS2_v2 --min-instances 0 --max-instances 1 \
    --idle-time-before-scale-down 120 --tier dedicated \
    --identity-type UserAssigned --user-assigned-identities "$IDS_CSV" \
    --tags "${TAGS[@]}" -o none
  echo "created $CLUSTER"
fi
az ml compute show -n "$CLUSTER" -g "$RG" -w "$WS" \
  --query "{size: size, tier: tier, min: min_instances, max: max_instances, idle_s: idle_time_before_scale_down}" -o tsv
az ml compute show -n "$CLUSTER" -g "$RG" -w "$WS" --query "identity.user_assigned_identities[].resource_id" -o tsv \
  | sed 's|.*/|  identity |'
echo "  workspace's own registry: $(az ml workspace show -n "$WS" -g "$RG" --query container_registry -o tsv | sed 's|.*/||')"

step "Registry: mark the version staging serves as production"
PROD="$(az ml model list -n "$MODEL_NAME" -g "$RG" -w "$WS" --query "[?tags.role=='production'].version" -o tsv)"
if [ -z "$PROD" ]; then
  az ml model update -n "$MODEL_NAME" --version 2 -g "$RG" -w "$WS" \
    --set tags.role=production tags.adopted=bootstrap -o none
  PROD=2
fi
echo "  $MODEL_NAME v$PROD is production (the gate's incumbent)"

step "cloud.env"
add() {  # key value
  if grep -qE "^$1=" cloud.env; then
    sed -i.bak "s|^$1=.*|$1=$2|" cloud.env && rm -f cloud.env.bak
  else
    printf '%s=%s\n' "$1" "$2" >> cloud.env
  fi
  echo "  $1=$2"
}
add TRAIN_INSTANCE "$CLUSTER"
add TRAIN_IDENTITY_CLIENT_ID "${CLIENT[itcs355-train]}"
add DEPLOY_IDENTITY_CLIENT_ID "${CLIENT[itcs355-github]}"
add SERVE_IDENTITY_CLIENT_ID "${CLIENT[itcs355-serve]}"
add CONTAINER_APP "$APP"
# The capstone's cluster is tagged lab=5 too; teardown must not take it with Lab 5.
add TEARDOWN_KEEP "itcs3556688020-cpu"

cat <<EOF

Done. Staging: https://$FQDN  (model v$PROD)

GitHub, for CI to log in again (Settings > Secrets and variables > Actions):
  secret   AZURE_CLIENT_ID = ${CLIENT[itcs355-github]}
  variable IDENTITY_REF    = ${CLIENT[itcs355-github]}
  variable STAGING_URL     = https://$FQDN

Next: build and push the training image, then submit the pipeline (README, Lab 5).
EOF
