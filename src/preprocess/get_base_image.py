import sagemaker


def main(framework, region, version, instance_type):
    return sagemaker.image_uris.retrieve(
        framework, region, version, instance_type=instance_type, py_version="py3", image_scope="training"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    aa = parser.add_argument
    aa("--framework", type=str, help="framework name")
    aa("--region", type=str, help="AWS region")
    aa("--version", type=str, help="framework version")
    aa("--instance-type", type=str, help="SageMaker instance type")

    args = parser.parse_args()

    print(main(args.framework, args.region, args.version, args.instance_type))
