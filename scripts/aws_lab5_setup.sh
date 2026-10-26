#!/usr/bin/env bash
# ITCS355 Lab 5 Task 4 — stand up the second provider: just enough AWS for the three
# seam methods (upload, download, invoke) to run for real.
#
#   aws configure                  # the itcs355-cli user's key (docs/aws-cli-policy.json)
#   bash scripts/aws_lab5_setup.sh
#
# Creates, all tagged course=itcs355 student=<PROJECT_ID> lab=5 and named itcs355-*:
#   S3 bucket       itcs355-<student>-<account>   BLOB_URI; holds the model file
#   ECR repository  itcs355                       the Lambda image
#   IAM role        itcs355-lambda-exec           the function's identity: read the model
#                                                 prefix, write its own logs, nothing else
#   Lambda          itcs355-predict               the SAME serving image as Azure, plus the
#                                                 Lambda Web Adapter (service/Dockerfile.lambda)
# and writes cloud.aws.env (gitignored) for scripts/portability_swap_check.py.
#
# The model it serves is v3's recipe (logistic regression on the week-44 batch), rebuilt
# inside the serving image so the scikit-learn version matches, uploaded to S3, and read
# back by the service at cold start through cloudlayer/aws.py's download().
#
# Region: ap-southeast-2 (Sydney). This account is on AWS's Free plan, which an
# AWS-managed service control policy locks to the region chosen at sign-up: CreateBucket
# was denied in ap-southeast-1 and us-east-1 and allowed in ap-southeast-2 (27 Oct).
#
# Cost: the free plan covers it. A few Lambda invocations, kilobytes in S3, one ~1 GB
# image in ECR (~0.10 USD per GB-month).
# Teardown: python scripts/teardown_verify.py --lab 5 --env cloud.aws.env
set -euo pipefail
export MSYS_NO_PATHCONV=1
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*)
    aws() { command aws "$@" | tr -d '\r'; }
    WINPWD="$(pwd -W)" ;;
  *) WINPWD="$(pwd)" ;;
esac

cd "$(dirname "$0")/.."
REGION="${AWS_REGION:-ap-southeast-2}"
STUDENT="$(tr -d '\r' < cloud.env | grep -E '^PROJECT_ID=' | cut -d= -f2-)"
MODEL_NAME="$(tr -d '\r' < cloud.env | grep -E '^MODEL_REGISTRY_NAME=' | cut -d= -f2-)"
ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
BUCKET="itcs355-${STUDENT#itcs355-}-${ACCOUNT}"
REPO=itcs355
ROLE=itcs355-lambda-exec
FUNCTION=itcs355-predict
REGISTRY="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${REPO}"
SHA="$(git rev-parse --short HEAD)"
TAGS_KV=(course=itcs355 "student=${STUDENT}" lab=5)
step() { printf '\n== %s\n' "$*"; }

echo "account $ACCOUNT  region $REGION  as $(aws sts get-caller-identity --query Arn --output text)"

step "S3 bucket $BUCKET"
if ! aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
  aws s3api create-bucket --bucket "$BUCKET" --region "$REGION" \
    --create-bucket-configuration "LocationConstraint=$REGION" >/dev/null
  aws s3api put-public-access-block --bucket "$BUCKET" --public-access-block-configuration \
    BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
fi
aws s3api put-bucket-tagging --bucket "$BUCKET" --tagging \
  "TagSet=[{Key=course,Value=itcs355},{Key=student,Value=${STUDENT}},{Key=lab,Value=5}]"
echo "  s3://$BUCKET (private)"

step "ECR repository $REPO"
if ! aws ecr describe-repositories --repository-names "$REPO" --region "$REGION" >/dev/null 2>&1; then
  aws ecr create-repository --repository-name "$REPO" --region "$REGION" \
    --tags Key=course,Value=itcs355 "Key=student,Value=${STUDENT}" Key=lab,Value=5 >/dev/null
fi
echo "  $REGISTRY"

step "Serving image (the Azure one, built locally) and the model it will serve"
docker build -q -f service/Dockerfile.serve -t itcs355-serve:local . >/dev/null
mkdir -p reports
docker run --rm -v "${WINPWD}:/work" -w /work --entrypoint sh itcs355-serve:local -c \
  "python scripts/make_dataset.py --seed 202644 --out /tmp/wk44.csv >/dev/null && \
   python scripts/export_model.py --model logreg --data /tmp/wk44.csv --out /work/reports/model-v3-aws.joblib"
aws s3 cp reports/model-v3-aws.joblib "s3://${BUCKET}/itcs355/models/v3/model.joblib" --only-show-errors
MODEL_URI="s3://${BUCKET}/itcs355/models/v3/model.joblib"
echo "  $MODEL_URI"

step "Lambda image: serving image + Lambda Web Adapter"
# --provenance=false: Lambda rejects the multi-manifest index Docker Desktop's default
# build attestations produce.
docker build -q --provenance=false -f service/Dockerfile.lambda --build-arg BASE=itcs355-serve:local \
  -t "itcs355-lambda:${SHA}" . >/dev/null
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "${REGISTRY%%/*}" >/dev/null
docker tag "itcs355-lambda:${SHA}" "${REGISTRY}:lambda-${SHA}"
docker push -q "${REGISTRY}:lambda-${SHA}" >/dev/null
DIGEST="$(aws ecr describe-images --repository-name "$REPO" --region "$REGION" \
  --image-ids "imageTag=lambda-${SHA}" --query 'imageDetails[0].imageDigest' --output text)"
IMAGE="${REGISTRY}@${DIGEST}"
echo "  $IMAGE"

step "Execution role $ROLE"
if ! aws iam get-role --role-name "$ROLE" >/dev/null 2>&1; then
  aws iam create-role --role-name "$ROLE" --assume-role-policy-document \
    '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}' \
    --tags Key=course,Value=itcs355 "Key=student,Value=${STUDENT}" Key=lab,Value=5 >/dev/null
  aws iam attach-role-policy --role-name "$ROLE" \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
  echo "  waiting 15 s for the new role to propagate"; sleep 15
fi
aws iam put-role-policy --role-name "$ROLE" --policy-name read-model --policy-document \
  "{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Action\":\"s3:GetObject\",\"Resource\":\"arn:aws:s3:::${BUCKET}/itcs355/models/*\"}]}"
ROLE_ARN="$(aws iam get-role --role-name "$ROLE" --query Role.Arn --output text)"
echo "  $ROLE_ARN: s3:GetObject on models/ only, plus its own logs"

step "Lambda function $FUNCTION"
ENV_VARS="Variables={CLOUD_PROVIDER=aws,PROJECT_ID=${STUDENT},REGION=${REGION},BLOB_URI=s3://${BUCKET}/itcs355,MODEL_URI=${MODEL_URI},MODEL_VERSION=3-aws,BUILD_SHA=${SHA},MODEL_REGISTRY_NAME=${MODEL_NAME}}"
if aws lambda get-function --function-name "$FUNCTION" --region "$REGION" >/dev/null 2>&1; then
  aws lambda update-function-code --function-name "$FUNCTION" --image-uri "$IMAGE" --region "$REGION" >/dev/null
  aws lambda wait function-updated-v2 --function-name "$FUNCTION" --region "$REGION"
  aws lambda update-function-configuration --function-name "$FUNCTION" --environment "$ENV_VARS" \
    --region "$REGION" >/dev/null
else
  aws lambda create-function --function-name "$FUNCTION" --package-type Image --code "ImageUri=${IMAGE}" \
    --role "$ROLE_ARN" --memory-size 1024 --timeout 60 --architectures x86_64 \
    --environment "$ENV_VARS" --region "$REGION" \
    --tags "course=itcs355,student=${STUDENT},lab=5" >/dev/null
fi
aws lambda wait function-active-v2 --function-name "$FUNCTION" --region "$REGION"
aws lambda wait function-updated-v2 --function-name "$FUNCTION" --region "$REGION"
echo "  active (1024 MB, 60 s timeout, private: no function URL)"

step "cloud.aws.env"
cat > cloud.aws.env <<EOF
# Second provider for Lab 5 Task 4. Written by scripts/aws_lab5_setup.sh. Gitignored.
CLOUD_PROVIDER=aws
PROJECT_ID=${STUDENT}
REGION=${REGION}
BLOB_URI=s3://${BUCKET}/itcs355
CONTAINER_REGISTRY=${REGISTRY}
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
MODEL_REGISTRY_NAME=${MODEL_NAME}
IDENTITY_REF=${ROLE_ARN}
SERVING_ENDPOINT=${FUNCTION}
EOF
echo "  wrote cloud.aws.env"

cat <<EOF

Done. Now run the swap check (the first invoke is a cold start, 20-40 s):
  .\\.venv-win\\Scripts\\python scripts\\portability_swap_check.py --second-provider aws --env cloud.aws.env
EOF
