#!/usr/bin/env bash
# ITCS355 Lab 5 Task 6 — tear down everything Lab 5 created, on both clouds.
#
#   bash scripts/lab5_teardown.sh            # lists what it will delete, asks, then deletes
#
# Azure, by name (not by tag: the capstone's cluster also carries lab=5):
#   the weekly retraining schedule      itcs355-retrain-weekly   (first: a schedule outliving
#                                                                 its cluster fails every week)
#   the Lab 5 compute cluster           itcs355-lab5-cpu
#   the staging Container App           itcs355-predict
#   the three identities                itcs355-train / -serve / -github, roles first
# AWS, everything tagged course=itcs355 lab=5 (cloudlayer/aws.py teardown) plus the
#   function's log group.
#
# Kept on purpose, and why:
#   registry, workspace, storage account, Container Apps environment: shared with the capstone
#   registered model versions v1-v3: records, not running resources; they cost nothing
#   the AWS CLI user itcs355-cli: it runs this script. Delete it in the console afterwards.
set -euo pipefail
export MSYS_NO_PATHCONV=1
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*)
    AZ_BIN="$(command -v az || command -v az.cmd || true)"
    az() { "$AZ_BIN" "$@" | tr -d '\r'; }
    aws() { command aws "$@" | tr -d '\r'; }
    PY=".venv-win/Scripts/python" ;;
  *) PY="python" ;;
esac
cd "$(dirname "$0")/.."
get() { tr -d '\r' < cloud.env | grep -E "^$1=" | head -1 | cut -d= -f2- | sed 's/[[:space:]]*#.*$//'; }
RG="$(get PROJECT_ID)"; WS="$(get AZURE_ML_WORKSPACE)"
APP=itcs355-predict; CLUSTER=itcs355-lab5-cpu; SCHEDULE=itcs355-retrain-weekly
IDENTITIES=(itcs355-train itcs355-serve itcs355-github)

echo "== Will delete on Azure ($RG)"
echo "  schedule  $SCHEDULE"
echo "  cluster   $CLUSTER"
echo "  app       $APP"
for i in "${IDENTITIES[@]}"; do echo "  identity  $i (and its role assignments)"; done
echo "== Will delete on AWS: everything tagged course=itcs355 lab=5, and /aws/lambda/itcs355-predict logs"
echo "== Will NOT touch: tagged lab=5 but kept: $(get TEARDOWN_KEEP); registry, workspace, storage, environment"
printf '\nType yes to continue: '
read -r answer
[ "$answer" = "yes" ] || { echo "nothing deleted"; exit 1; }

echo; echo "== Azure"
"$PY" -m cloudlayer.pipelines unschedule || echo "  (schedule: see message above)"
if az ml compute show -n "$CLUSTER" -g "$RG" -w "$WS" -o none 2>/dev/null; then
  az ml compute delete -n "$CLUSTER" -g "$RG" -w "$WS" --yes -o none && echo "  deleted cluster $CLUSTER"
fi
if az containerapp show -n "$APP" -g "$RG" -o none 2>/dev/null; then
  az containerapp delete -n "$APP" -g "$RG" --yes -o none && echo "  deleted app $APP"
fi
for i in "${IDENTITIES[@]}"; do
  if az identity show -n "$i" -g "$RG" -o none 2>/dev/null; then
    pid="$(az identity show -n "$i" -g "$RG" --query principalId -o tsv)"
    az role assignment delete --assignee "$pid" -o none 2>/dev/null || true
    az identity delete -n "$i" -g "$RG" -o none && echo "  deleted identity $i"
  fi
done
JOIN_ROLE="ITCS355 Container Apps environment join"
if [ -n "$(az role definition list --name "$JOIN_ROLE" --query "[].id" -o tsv)" ]; then
  az role definition delete --name "$JOIN_ROLE" -o none && echo "  deleted custom role $JOIN_ROLE"
fi

echo; echo "== AWS"
if [ -f cloud.aws.env ]; then
  REGION="$(tr -d '\r' < cloud.aws.env | grep -E '^REGION=' | cut -d= -f2-)"
  "$PY" scripts/teardown_verify.py --lab 5 --env cloud.aws.env || true
  aws logs delete-log-group --log-group-name /aws/lambda/itcs355-predict --region "$REGION" 2>/dev/null \
    && echo "  deleted log group /aws/lambda/itcs355-predict" || echo "  (no log group left)"
else
  echo "  cloud.aws.env not found: skipped"
fi

cat <<EOF

Deletion is asynchronous. Re-check now, and again in 24 hours:
  az resource list -g $RG --tag lab=5 -o table          # only $(get TEARDOWN_KEEP) should remain
  az ml compute list -g $RG -w $WS -o table              # no itcs355-lab5-cpu
  az ml schedule list -g $RG -w $WS -o table             # empty
  $PY scripts/teardown_verify.py --lab 5 --env cloud.aws.env   # PASS nothing found
Then delete the IAM user itcs355-cli and the policy itcs355-cli-policy in the AWS console,
and screenshot the empty resource lists.
EOF
