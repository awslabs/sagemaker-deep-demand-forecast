#!/bin/bash

if [ $# -lt 3 ]; then
    echo "Please provide the solution name as well as the base S3 bucket name and the region to run build script."
    exit 1
fi

# Upload to S3
s3_prefix="s3://$2-$3/$1"
echo "Using S3 path: $s3_prefix"
aws s3 cp --recursive src $s3_prefix/src
aws s3 cp --recursive deploy $s3_prefix/deploy
