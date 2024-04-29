# ECO-395-Project-Somin-Lee-Chenxin-zhu-Tianyu-Wang
The project, co-authored by Somin Lee, Chenxin zhu and Tianyu Wang, aims to explore issues related to low fertility in South Korea. We analyzed the datasets and built models for comparison, and finally obtained a model with a good fit degree and analyzed the contribution of different variables to the model fit. Finally, the variables with high contribution degree are analyzed according to their practical significance, and their practical logic is found.

Reproduction

We used two different languages to complete the project: R and python. 

For R part, please follow the [Project_0429.Rmd](./Project_0429.Rmd) file in repository and use the dataset [Data.csv](./data/Data.csv) in data folder. The packages used are as follow:
```
library(readxl)
library(dplyr)
library(ggplot2)
library(caret)
library(randomForest)
library(e1071)
library(factoextra)
library(readr)
library(rpart)
library(gbm)
library(readr)
library(osmdata)
library(tidyverse)
library(tidyr)
library(rsample) 
library(ggplot2)
library(pROC)
```



For python part, please run [machinelearning_final.ipynb](./Neuralnetwork/machinelearning_final.ipynb) using the [projectdata_new.csv](./data/projectdata_new.csv) in data folder. We have induced neural network to analyze and fit the data, and finally selected the top variables to combine with the variables selected by the traditional model for the final fit. We used the Pytorch framework, CUDA version 12.4.1, and the package is as follows:
```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from torch.utils.data import DataLoader, TensorDataset
import torch
import chardet
import torch.nn as nn
import torch.optim as optim
import shap
```
After importing packages, we cleaned the dataset. The NA values are populated as Missing and -999 for categorical value and numerical value seperately.
```
NUMERICAL_FILL = -999
CATEGORICAL_FILL = "Missing"

# Impute missing values with different constants based on data type
for col in df.columns:
    if df[col].dtype in ['int64', 'float64']:  # For numerical columns
        df[col].fillna(NUMERICAL_FILL, inplace=True)
    elif df[col].dtype == 'object':  # For categorical columns
        df[col].fillna(CATEGORICAL_FILL, inplace=True)
```
Then we dropped some variables in order to reduce distractions and build the neural network framework then build evaluation function to test the model.
```
# Split features and target
features = df.drop(['n_under_age10', 'n_fam_members', 'region', 'GENDER_RESP', 'AGE_RESP', 'HOUSE_TYPE', 'HOUSE_TYPE2', 'working_status', 'reason_not_working', 'working_field', 'unique_identifier', 'REASON_TIRED2', ], axis=1)
target = df['n_under_age10']
```

When we started testing the neural network, we found that the results were too random because the dataset was too small, so we decided to calculate the average contribution of multiple runs.
```
# Initialize SHAP explainer with the model prediction function and a sample of the transformed data
explainer = shap.KernelExplainer(f, X_transformed)

# Calculate SHAP values for the transformed data
shap_values = explainer.shap_values(X_transformed)

assert shap_values.shape[1] == X_transformed.shape[1], "Mismatch in number of SHAP values and features"
expected_value = explainer.expected_value
if isinstance(expected_value, np.ndarray):
    expected_value = expected_value[0] 
shap.decision_plot(expected_value, shap_values, X_transformed[:100], feature_names=feature_names)
```

Ultimately, we used SHAP value and decision plot to display the average value after running 100 times neural network and their contributions to final output.
