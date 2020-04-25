# Deep Demand Forecasting with Amazon SageMaker

This project provides an end-to-end solution for **Demand Forecasting** task using a new state-of-the-art *Deep Learning* model [LSTNet](https://arxiv.org/abs/1703.07015) available in [gluonts](https://github.com/awslabs/gluon-ts) and [Amazon SageMaker](https://aws.amazon.com/sagemaker/).

## Demand Forecasting

Demand forecasting deals with time-series data targeting various resource demands to streamline the decision-making process across businesses. Examples include predicting the number of

* Product sales across multiple regions in the next quarter
* Electricity consumption for multiple regions in next week
* AWS cloud servers usage for next day for a video streaming service
* Customer representatives to hire for multiple locations in next month

## Why Deep Learning methods?

The status quo approaches for time-series forecasting are auto-regressive methods such as [Auto Regressive Integrated Moving Average](https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average) (ARIMA), [Box-Jenkins](https://en.wikipedia.org/wiki/Box%E2%80%93Jenkins_method) and *State-Space models* for **uni-variate time-series** data and [Vector Autoregression](https://en.wikipedia.org/wiki/Vector_autoregression) (VAR) or Gaussian Processing (GP) non-parametric models for **multi-variant** time-series data. The use of Deep Learning (DL) models for (stationary/non-stationary) multi-variate time-series has been a point of research recently. DL methods shine when dealing with *large* number of (correlated) multivariate time-series data that have *categorical features* and (a lot of) *missing values*. Neural network models can predict seasonality for new events since these global models learn patterns *jointly* over the whole dataset and can extrapolate these learned regularities to new series.

On the other hand, it is known that for classical methods, tedious data preprocessing and features generation need to be performed prior to model training and one main advantage of DL methods such as LSTNet is automating the feature generation step with better prediction power and fast GPU-enabled training and deployment.

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
*  Once the stack was created, click on the **SageMakerNotebookInstanceSignOn** link on the **Outputs tab**. 
*  Finally, click on **deep-demand-forecast.ipynb** notebook and follow the instruction inside the notebook.





Alternatively, you can clone this repository then navigate to [AWS CloudFormation](https://aws.amazon.com/cloudformation/) in your account and use the provided [CloudFormation template](deploy/sagemaker-deep-demand-forecast.yaml) to create the AWS resources needed to train and deploy the model using the SageMaker [deep-demand-forecast](src/deep-demand-forecast.ipynb) notebook.

## What does `deep-demand-forecast.ipynb` offer?

The notebook trains an [LSTNet](https://gluon-ts.s3-accelerate.dualstack.amazonaws.com/master/api/gluonts/gluonts.model.lstnet.html) estimator *on electricity consumption* data (for 5 epochs, for example) and we can compare its performance by visualizing the metrics [MASE](https://en.wikipedia.org/wiki/Mean_absolute_scaled_error) vs. [sMAPE](https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error)

<p align="center">
  <img src="docs/MASE_vs_sMAPE.gif" alt="MASE vs. sMAPE" width="300" height="250"/>
  <img src="docs/MASE_vs_sMAPE_barplots.png" alt="MASE sMAPE barplots" width="500" height="250"/>
</p>

Finally, we deploy an endpoint for the trained model and can interactively compare its performance by comparing the train, test data and predictions.

<p align="center">
  <img src="docs/interactive_plots.gif" alt=interative" width="500" height="400"/>
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
