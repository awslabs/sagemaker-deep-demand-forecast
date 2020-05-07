# Deep Demand Forecasting with Amazon SageMaker

This project provides an end-to-end solution for **Demand Forecasting** task using a new state-of-the-art *Deep Learning* model [LSTNet](https://arxiv.org/abs/1703.07015) available in [GluonTS](https://github.com/awslabs/gluon-ts) and [Amazon SageMaker](https://aws.amazon.com/sagemaker/).

## Demand Forecasting

Demand forecasting uses historical time-series data to help streamline the supply-demand decision-making process across businesses. Examples include predicting the number of

* Customer representatives to hire for multiple locations in the next month
* Product sales across multiple regions in the next quarter
* Cloud server usage for next day for a video streaming service
* Electricity consumption for multiple regions over the next week
* IoT devices and sensors such as energy consumption

## Deep Learning for time-series forecasting

The most used approaches for time-series forecasting are auto-regressive methods such as [Auto Regressive Integrated Moving Average](https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average) (ARIMA) for **uni-variate time-series** data and [Vector Autoregression](https://en.wikipedia.org/wiki/Vector_autoregression) (VAR) for **multi-variate** time-series data. These methods often require tedious data preprocessing and features generation prior to model training. One main advantage of Deep Learning (DL) methods such as LSTNet is automating the feature generation step with better prediction power and fast GPU-enabled training and deployment. DL methods shine when dealing with *large* number of (correlated) multi-variate time-series data that have *categorical features* and (a lot of) *missing values*. Neural network models can predict seasonality for new events since these global models learn patterns *jointly* over the whole dataset and can better extrapolate these learned regularities to new series.

## Getting started

You will need an AWS account to use this solution. Sign up for an account [here](https://aws.amazon.com/).

The easiest is to click on the following button to create the *AWS CloudFormation Stack* required for this solution

<table align="center">
  <tr>
    <th colspan="3">AWS Region</td>
    <th>AWS CloudFormation</td>
  </tr>
  <tr>
    <td>US West</td>
    <td>Oregon</td>
    <td>us-west-2</td>
    <td align="center">
      <a href="https://us-west-2.console.aws.amazon.com/cloudformation/home?region=us-west-2#/stacks/create/review?templateURL=https://sagemaker-solutions-us-west-2.s3-us-west-2.amazonaws.com/sagemaker-deep-demand-forecast/deploy/sagemaker-deep-demand-forecast.yaml&stackName=sagemaker-deep-demand-forecast&param_SolutionS3BucketName=sagemaker-ddf">
        <img src="docs/launch.svg" height="35">
      </a>
    </td>
  </tr>
</table>

Then acknowledge adding the default [AWS IAM policy](https://aws.amazon.com/iam/) or use your own policy

*  Click on the **Create Stack** (you can leave the pre-specified *Stack name*, *S3 Bucket Name* and SageMaker Notebook Instance as they are)
*  Once the stack was created, go to the **Outputs tab** and click on the **SageMakerNotebookInstanceSignOn** link to directly go to the created notebook instance 
*  Finally, click on **deep-demand-forecast.ipynb** notebook and follow the instruction inside the notebook


Alternatively, you can clone this repository then navigate to [AWS CloudFormation](https://aws.amazon.com/cloudformation/) in your account and use the provided [CloudFormation template](deploy/sagemaker-deep-demand-forecast.yaml) to create the AWS resources needed to train and deploy the model using the SageMaker [deep-demand-forecast](src/deep-demand-forecast.ipynb) notebook.

## What does `deep-demand-forecast.ipynb` offer?

The notebook trains an [LSTNet](https://gluon-ts.s3-accelerate.dualstack.amazonaws.com/master/api/gluonts/gluonts.model.lstnet.html) estimator *on electricity consumption* data (for 5 epochs, for example) and we can compare its performance by visualizing the metrics [MASE](https://en.wikipedia.org/wiki/Mean_absolute_scaled_error) vs. [sMAPE](https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error)

<p align="center">
  <img src="docs/MASE_vs_sMAPE.gif" alt="MASE vs. sMAPE" width="350" height="300"/>
  <img src="docs/MASE_sMAPE_barplots.png" alt="MASE sMAPE barplots" width="600" height="300"/>
</p>

Finally, we deploy an endpoint for the trained model and can interactively compare its performance by comparing the train, test data and predictions.

<p align="center">
  <img src="docs/interactive_plots.gif" alt=interative" width="700" height="600"/>
</p>

## Architecture overview

The project contains

* [**Preprocessing**](src/preprocess) step, designed as a *microservice* that allows users to build and register their own Docker image for this task via [Amazon ECR](https://aws.amazon.com/ecr/) and execute the job in [Amazon SageMaker](https://aws.amazon.com/sagemaker/)
* Interactive **training**, **evaluating** and **visualizing** the results in the provided [SageMaker notebook](source/deep_demand_forecast.ipynb)
* **Deplying** and **testing** an [HTTPS endpoint](https://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-hosting.html)
* Monitoring the deployed model via [Amazon CloudWatch](https://aws.amazon.com/cloudwatch/)

Here is the visual architecture

<p align="center">
  <img src="docs/arch.png" alt="Solution Architecture" width="600" height="500">
</p>

## License

This project is licensed under the Apache-2.0 License.
