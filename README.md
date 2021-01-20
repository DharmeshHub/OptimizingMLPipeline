# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model. I used HyperDrive to optimise the hyperparameter of the model. Additionally used Azure AutoML to find the optimal model using same dataset and compare the resultes of the both the methods.

## Summary
**In 1-2 sentences, explain the problem statement: e.g "This dataset contains data about... we seek to predict..."**

The data used here is of direct marketing campaigns through phone calls for banking institution. The classification goal is to predict if customer will subscribe to term deposit (yes/no). Dataset consits of 32950 records with 20 independent variables out of which 10 are numeric features and 10 are categorical features. Additionally targetvariable is "y".

**In 1-2 sentences, explain the solution: e.g. "The best performing model was a ..."**
The best performing model was the HyperDrive model with ID <HD_fda34223-a94c-456b-8bf7-52e84aa1d17e_14. It derived from a Scikit-learn pipeline and had an accuracy of 0.91760. 
In contrast, for the AutoML model with ID AutoML_ee4a685e-34f2-4031-a4f9-fe96ff33836c_13, the accuracy was 0.91618 and the algorithm used was VotingEnsemble.


## Scikit-learn Pipeline
**Explain the pipeline architecture, including data, hyperparameter tuning, and classification algorithm.**

![Pipeline diagram](/Images/Pipeline.jfif)

Pipeline architecture:<br/>
		Dataset - Created the dataset using TabularDatasetFactory. Split data into train and test (0.2) sets.<br/>
		Train data - Created the training script to train the marketing data dataset using Scikit-learn Logistic regression algorithm.<br/>
		HyperDrive - Used HyperDrive with specified parameter sampler and policy for early stopping to find the optimal hyperparameter for logistic regression model. This will give us trained model with optimize hyperparameter.<br/>
	

**What are the benefits of the parameter sampler you chose?**

I chose Random Paramer Sampler:
	Random sampling supports discrete and continuous hyperparameters. It supports early termination of low-performance runs. It is computationally less expensive as it takes subset of combinations and it's faster unlike GridParameterSampling. Some users do an initial search with random sampling and then refine the search space to improve results. In random sampling, hyperparameter values are randomly selected from the defined search space.
	GridParameterSampling utilize more resources compare to RandomParameterSampling.


RandomParameter Sampling is computationally less expensive as it just takes a subset of combinations. On the other hand GridParameterSampling takes all the possible combinations of the hyperparameters and hence is computationally very expensive. For the future iteration, you may also try out BayesianParameter Sampling Technique which intelligently picks the next sample of hyperparameters, based on how the previous samples performed, such that the new sample improves the reported primary metric. It was the right choice to use BanditPolicy as it cancels the runs which are not up to the mark !!

Random sweep: When you select this option, the module will randomly select parameter values over a system-defined range. You must specify the maximum number of runs that you want the module to execute. This option is useful when you want to increase model performance by using the metrics of your choice but still conserve computing resources.

ps = RandomParameterSampling( 
    {
        "--max_iter": choice(10,50,100,150,200)
        ,"--C": choice(0.5,0.8,0.9,1,1.25,1.5)
    }
)

Here i chose discrete values with _choice_ for both parameters, _C_ and _max_iter_. _C_ is the Regularization and _max_iter_ is the maximum number of iterations. 
This option trains a model by using a set number of iterations. You specify a range of values to iterate over, and the module uses a randomly chosen subset of those values. Values are chosen with replacement, meaning that numbers previously chosen at random are not removed from the pool of available numbers. So the chance of any value being selected stays the same across all passes.


**What are the benefits of the early stopping policy you chose?**

Early stopping policy automatically terminates poorly performing runs. For more aggressive savings, used Bandit Policy with a smaller allowable slack or Truncation Selection Policy with a larger truncation percentage. Any run that doesn't fall within the slack factor or slack amount of the evaluation metric with respect to the best performing run will be terminated. This means that with this policy, the best performing runs will execute until they finish.

policy = BanditPolicy(evaluation_interval=1, slack_factor=0.1, delay_evaluation=5)

_evaluation_interval_: (optional) the frequency for applying the policy. Each time the training script logs the primary metric counts as one interval.

_slack_factor_: The amount of slack allowed with respect to the best performing training run. This factor specifies the slack as a ratio.

_delay_evaluation: (optional) delays the first policy evaluation for a specified number of intervals.


## AutoML
**In 1-2 sentences, describe the model and hyperparameters generated by AutoML.**

Auto ML:
	Dataset - Created Tabular dataset.
	Auto ML - Used Auto ML to find best training model.
	
Below is the AutoML configuration i set for this project.

automl_config = AutoMLConfig(
    compute_target=compute_target,
    experiment_timeout_minutes=20,
    task='classification',
    primary_metric="accuracy",
    training_data=train_data,
    label_column_name="y",
    n_cross_validations=5
)

_experiment_timeout_minutes=20_

This is an exit criterion and is used to define how long (in minutes), the experiment should continue to run. To help avoid experiment time out failures, I used the minimum of 20 minutes.

_task='classification'_

This defines the experiment type which in this case is classification.

_primary_metric='accuracy'_

I chose accuracy as the primary metric for this classification model.

_n_cross_validations=5_

This parameter sets how many cross validations to perform, based on the same number of folds (subsets). Five folds for cross-validation are defined. So, five different trainings, each training using 4/5 of the data, and each validation using 1/5 of the data with a different holdout fold each time.


## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**

run_id - HD_6606dab1-d34f-429d-8535-6b44ff6a8ab9_29
Accuracy - 0.9078907435508345
Parameter sampling - Random
Termination Policy - BANDIT


run_id = AutoML_f6431fa8-cd51-40d7-817e-97ca693a2e1d_4
Accuracy - 0.915585873177552
Algorithm - VotingEnsemble

## Future work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**

Entire grid: 


## Proof of cluster clean up
**If you did not delete your compute cluster in the code, please complete this section. Otherwise, delete this section.**
**Image of cluster marked for deletion**
