#!/usr/bin/env bash

image=$1

region=$2

account=$3

fullname="${account}.dkr.ecr.${region}.amazonaws.com/${image}:latest"

export BASE_IMAGE=$(python get_base_image.py --framework="mxnet" --region=${region} \
    --version="1.6.0" --instance-type="ml.m5.xlarge")

aws ecr describe-repositories --repository-names ${image} --region ${region} >/dev/null 2>&1

if [ $? -ne 0 ]; then
    aws ecr create-repository --repository-name ${image} --region ${region} >/dev/null
fi

SAGEMAKER_ACCOUNT=$(echo ${BASE_IMAGE} | cut -c 1-12)

# Get the login command from ECR and execute it directly for both accounts
aws ecr get-login-password --region ${region} | docker login \
    --username AWS \
    --password-stdin ${account}.dkr.ecr.${region}.amazonaws.com

aws ecr get-login-password --region ${region} | docker login \
    --username AWS \
    --password-stdin ${SAGEMAKER_ACCOUNT}.dkr.ecr.${region}.amazonaws.com

# Build the image and then push it to ECR with the full name
docker build -t ${image} --build-arg BASE_IMAGE=${BASE_IMAGE} container/codebuild
docker tag ${image} ${fullname}
docker push ${fullname}
