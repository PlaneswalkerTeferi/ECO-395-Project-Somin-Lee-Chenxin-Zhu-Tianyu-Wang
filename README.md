# ECO-395-Project-Somin-Lee-Chenxin-zhu-Tianyu-Wang
The project, co-authored by Somin Lee, Chenxin zhu and Tianyu Wang, aims to explore issues related to low fertility in South Korea. We analyzed the datasets and built models for comparison, and finally obtained a model with a good fit degree and analyzed the contribution of different variables to the model fit. Finally, the variables with high contribution degree are analyzed according to their practical significance, and their practical logic is found.

Reproduction

We used two different languages to complete the project: R and python. 

For R part, please follow the [Project_0429.Rmd](./data/Project_0429.Rmd) file in repository and use the dataset [Data.csv](./data/Data.csv) in data folder.


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
After importing packages, we cleaned the dataset. The NA values are populated as Missing and -999 for categorical value and numerical value seperately. Then we dropped some variables in order to reduce distractions and build the neural network framework then build evaluation function to test the model.
Ultimately, we used SHAP value and decision plot to display the average value after running 100 times neural network and their contributions to final output.
