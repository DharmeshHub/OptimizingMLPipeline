# Udacity Capstone Project - Using Machine Learning to Predict Survival of Patients with Heart Failure


In this project, we have created two models: one using Automated ML (AutoML) and one customized model whose hyperparameters are tuned using HyperDrive. We then compare the performance of both the models and deploy the best performing model. We have used Patient Heart failure data to predict survival, source of data is from publicly available <a href="https://www.kaggle.com">Kaggle</a> dataset. We consume the deployed model using rest endpoint URL. 

Below is overall task flow of this project:

![Project Workflow](images/Capstone_Project.jpg?raw=true "Project Workflow") 

## Project Set Up and Installation
This is project used Azure Machine Learning studio provided by Udacity Lab. We have used following Jupyter notebook files to completes this project.

In order to run the project in Azure Machine Learning Studio, we will need the two Jupyter Notebooks files, dataset file and other supporting files to build models.

- `automl.ipynb`: AutoML model;
- `hyperparameter_tuning.ipynb`: HyperDrive model.
- `heart_failure_clinical_records_dataset.csv`: the dataset file.

- Additionally, Models are created under Python 3.6 SDK environment and xgboost with verion 1.3.3 was installed. I removed version 1.3.3 and installed version xgboost==0.90 version. This we need to get best run detail detail.

## Dataset

### Overview
*TODO*: Explain about the data you are using and where you got it from.

Cardiovascular diseases (CVDs) are the number one of the most cause of death globally, taking an estimated 17.9 million lives each year.
Heart failure is a common event caused by CVDs and this dataset contains 12 features that can be used to predict mortality by heart failure.
Most cardiovascular diseases can be prevented by addressing behavioural risk factors such as tobacco use, unhealthy diet and obesity, physical inactivity and harmful use of alcohol using population-wide strategies.
People with cardiovascular disease or who are at high cardiovascular risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyperlipidaemia or already established disease) need early detection and management wherein a machine learning model can be of great help.

This dataset has been taken from kaggle site - https://www.kaggle.com/andrewmvd/heart-failure-clinical-data?select=heart_failure_clinical_records_dataset.csv

Below are (13) clinical features in this dataset:

- age: age of the patient (years)
- anaemia: decrease of red blood cells or hemoglobin (boolean)
- high blood pressure: if the patient has hypertension (boolean)
- creatinine phosphokinase (CPK): level of the CPK enzyme in the blood (mcg/L)
- diabetes: if the patient has diabetes (boolean)
- ejection fraction: percentage of blood leaving the heart at each contraction (percentage)
- platelets: platelets in the blood (kiloplatelets/mL)
- sex: woman or man (binary)
- serum creatinine: level of serum creatinine in the blood (mg/dL)
- serum sodium: level of serum sodium in the blood (mEq/L)
- smoking: if the patient smokes or not (boolean)
- time: follow-up period (days)
- [target] death event: if the patient deceased during the follow-up period (boolean)

![Dataset details](images/Dataset_detail.png?raw=true "Dataset detail") 


### Task
The model we build using Heart failure dataset classify whether patient have chance of survival or not. Model use above list features and result in target column (Death event) with value 0 (no) or 1 (yes).


### Access

I downloaded dataset file (.csv) from Kaggle site and added to my GitHub respository. I access dataset directly from my GitHub repository and registered the dataset into workspace. I have also added dataset file in my Jupyter notebook directory to access data during training (train.py)

***Create Dataset:*** _URI source_
[Create Dataset Registered](images/CreateDataset.png?raw=true "Create Dataset") 

***Registered datasets:*** _heart-failure-prediction_
![Dataset Registered](images/RegisteredDataset.png?raw=true "Dataset Registered") 

***Registered datasets Source:*** _URI_
![Dataset URI](images/RegisteredDatasetURI.png?raw=true "Dataset URI") 

 
## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

Below is the AutoML setting and configuration for this project.
![AutoML Configuration](images/AutoMLConfiguration.png?raw=true "AutoML Configuration") 

_experiment_timeout_minutes=20_

This is an exit criterion and is used to define how long (in minutes), the experiment should continue to run. To help avoid experiment time out failures, I used the minimum of 20 minutes.

_max_concurrent_iterations_: 4

It represents the maximum number of iterations that would be executed in parallel.

_task='classification'_

This defines the experiment type which in this case is classification.

_primary_metric='accuracy'_

I chose accuracy as the primary metric for this classification model.

_n_cross_validations=2_

This parameter sets how many cross validations to perform, based on the same number of folds (subsets). Two folds for cross-validation are defined. So, two different trainings, each training using 1/2 of the data, and each validation using 1/2 of the data with a different holdout fold each time.

_enable_early_stopping=True_

Early stopping helps in performance, it terminates poor performing run and fully run good performing run.

_featurization=auto_

Featurization is done automatically, i.e. normalization technique are applied to your data. This help certain algorithms that are sensitive to features on different scales.

_lable_column_name = "DEATH_EVENT"
The name of the label (target) column. This parameter is applicable to training_data and validation_data parameters.

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

Below is the best model run for AutoML model.
run_id = AutoML_c6507e5f-cd46-4860-a977-7732236037f7_16
Accuracy - 0.852841163310962
Algorithm - VotingEnsemble


Following are screenshot of AutoML experiment run.

***AutoML Completed:***
[AutoML Completed](images/AutoML_Exp_Completed.png?raw=true "AutoML Completed") 

***Data Guardrials:***
[Data Guardrials](images/DataGuardrials.png?raw=true "Data Guardrials") 

***AutoML Best Model:***
[AutoML Best Model](images/AutoML_Exp_BestModel1.png?raw=true "AutoML Best Model") 

***AutoML RunDetails Widget:***
[AutoML RunDetails](images/AutoML_Exp_Completed_SDK.png?raw=true "AutoML RunDetails") 

***AutoML runs:***
[AutoML Runs](images/Auto_RunDetails.png.png?raw=true "AutoML Runs") 


***Best model top features:***
[Best Model features](images/BestModel_Topfeatures.png.png?raw=true "Best Model features")

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

***Best model run details:*** _Details of Best run:_

[Best Runs](images/BestRunDetails.png.png?raw=true "Best Run") 

[Best Runs2](images/BestRunDetails1.png.png?raw=true "Best Run1") 

[Best Runs3](images/BestRunDetails2.png.png?raw=true "Best Run2")



## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
