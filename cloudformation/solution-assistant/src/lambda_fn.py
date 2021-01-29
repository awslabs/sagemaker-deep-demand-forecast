import boto3
import sys

sys.path.append("./site-packages")
from crhelper import CfnResource

helper = CfnResource()


@helper.create
def on_create(_, __):
    pass


@helper.update
def on_update(_, __):
    pass


def delete_sagemaker_endpoint(endpoint_name):
    sagemaker_client = boto3.client("sagemaker")
    try:
        sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
        print("Successfully deleted endpoint " "called '{}'.".format(endpoint_name))
    except sagemaker_client.exceptions.ClientError as e:
        if "Could not find endpoint" in str(e):
            print("Could not find endpoint called '{}'. " "Skipping delete.".format(endpoint_name))
        else:
            raise e


def delete_sagemaker_endpoint_config(endpoint_config_name):
    sagemaker_client = boto3.client("sagemaker")
    try:
        sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
        print(
            "Successfully deleted endpoint configuration "
            "called '{}'.".format(endpoint_config_name)
        )
    except sagemaker_client.exceptions.ClientError as e:
        if "Could not find endpoint configuration" in str(e):
            print(
                "Could not find endpoint configuration called '{}'. "
                "Skipping delete.".format(endpoint_config_name)
            )
        else:
            raise e


def delete_sagemaker_model(model_name):
    sagemaker_client = boto3.client("sagemaker")
    try:
        sagemaker_client.delete_model(ModelName=model_name)
        print("Successfully deleted model called '{}'.".format(model_name))
    except sagemaker_client.exceptions.ClientError as e:
        if "Could not find model" in str(e):
            print("Could not find model called '{}'. " "Skipping delete.".format(model_name))
        else:
            raise e


def delete_s3_objects(bucket_name):
    s3_resource = boto3.resource("s3")
    try:
        s3_resource.Bucket(bucket_name).objects.all().delete()
        print("Successfully deleted objects in bucket " "called '{}'.".format(bucket_name))
    except s3_resource.meta.client.exceptions.NoSuchBucket:
        print("Could not find bucket called '{}'. " "Skipping delete.".format(bucket_name))


def delete_s3_bucket(bucket_name):
    s3_resource = boto3.resource("s3")
    try:
        s3_resource.Bucket(bucket_name).delete()
        print("Successfully deleted bucket " "called '{}'.".format(bucket_name))
    except s3_resource.meta.client.exceptions.NoSuchBucket:
        print("Could not find bucket called '{}'. " "Skipping delete.".format(bucket_name))


def delete_ecr_images(repository_name):
    ecr_client = boto3.client("ecr")
    try:
        images = ecr_client.describe_images(repositoryName=repository_name)
        image_details = images["imageDetails"]
        if len(image_details) > 0:
            image_ids = [{"imageDigest": i["imageDigest"]} for i in image_details]
            ecr_client.batch_delete_image(repositoryName=repository_name, imageIds=image_ids)
            print(
                "Successfully deleted {} images from repository "
                "called '{}'. ".format(len(image_details), repository_name)
            )
        else:
            print(
                "Could not find any images in repository "
                "called '{}' not found. "
                "Skipping delete.".format(repository_name)
            )
    except ecr_client.exceptions.RepositoryNotFoundException:
        print(
            "Could not find repository called '{}' not found. "
            "Skipping delete.".format(repository_name)
        )


@helper.delete
def on_delete(event, __):
    # remove sagemaker endpoints
    solution_prefix = event["ResourceProperties"]["SolutionPrefix"]
    endpoint_names = [
        "{}-endpoint".format(solution_prefix),  # make sure it is the same as your endpoint name
    ]
    for endpoint_name in endpoint_names:
        delete_sagemaker_model(endpoint_name)
        delete_sagemaker_endpoint_config(endpoint_name)
        delete_sagemaker_endpoint(endpoint_name)

    # remove files in s3
    output_bucket = event["ResourceProperties"]["S3BucketName"]
    delete_s3_objects(output_bucket)

    # delete buckets
    delete_s3_bucket(output_bucket)

    # delete images from ecr
    ecr_repository = event["ResourceProperties"]["ECRRepository"]
    delete_ecr_images(ecr_repository)


def handler(event, context):
    helper(event, context)
