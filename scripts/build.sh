#!/bin/bash
set -euxo pipefail

NOW=$(date +"%x %r %Z")
echo "Time: $NOW"

if [ $# -lt 3 ]; then
    echo "Please provide the solution name as well as the base S3 bucket name and the region to run build script."
    echo "For example: ./scripts/build.sh trademarked-solution-name sagemaker-solutions-build us-west-2"
    exit 1
fi

DATASET="electricity"
BASE_DIR="$(dirname $(dirname $(readlink -f $0)))"
echo "Base dir: $BASE_DIR"

rm -rf build
mkdir build

for nb in src/*.ipynb; do
    python $BASE_DIR/scripts/set_kernelspec.py --notebook $nb --display-name "Python 3 (MXNet JumpStart)" --kernel "HUB_1P_IMAGE"
done

echo "Solution assistant lambda function"
cd cloudformation/solution-assistant/
python -m pip install -r requirements.txt -t ./src/site-packages

cd -

echo "Clean up pyc files, needed to avoid security issues. See: https://blog.jse.li/posts/pyc/"
find cloudformation/ -type f -name "*.pyc" -delete
find cloudformation/ -type d -name "__pycache__" -delete
cp -r cloudformation/solution-assistant/src build/
cd build/src
zip -q -r9 $BASE_DIR/build/solution-assistant.zip -- *

cd -
rm -rf build/src

echo "Copying and packaging data preprocessing container build source"
cp -r src/preprocess/ build/preprocess/
cd build/preprocess
zip -q -r9 $BASE_DIR/build/sagemaker-deep-demand-forecast-preprocessing.zip -- *

cd -
rm -rf build/preprocess

s3_prefix="s3://$2-$3/$1"

echo "Removing the existing objects under $s3_prefix"
aws s3 rm --recursive $s3_prefix
echo "Copying new objects to $s3_prefix"
aws s3 cp --recursive . $s3_prefix/

echo "Copying solution artifacts"
aws s3 cp "s3://sagemaker-solutions-artifacts/$1/model.tar.gz" $s3_prefix/build/model.tar.gz
aws s3 sync "s3://sagemaker-solutions-artifacts/$1/data/$DATASET" $s3_prefix/data/$DATASET
