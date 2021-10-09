# ML-Engineer_Capstone-Project

This is my Udacity Machine Learning Engineer Capstone Project

## Libaries and other prerequisites

```python

import os
import time
from time import gmtime, strftime
import pandas as pd
import numpy as np
import math
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import sagemaker
from sklearn.model_selection import train_test_split 
from sagemaker import LinearLearner
from sagemaker import get_execution_role
from sklearn.metrics import accuracy_score
from sagemaker.tuner import IntegerParameter, ContinuousParameter, HyperparameterTuner
import torch
import torch.nn as nn
import torch.nn.functional as F
```

An aws account is necessary to tun the machine learning models.

## Order for Notebooks
 
There are several notebooks for better organization.

1. Starbucks_Capstone_notebook.ipynb gives a small project introduction 
2. DataCleaning.ipynb creates the cleaned files and a merged full dataframe. All dataframe here created are written to csv files in data directory
3. Labeling.ipynb creates a dataframe with offers and the information for each event about offer viewed and offer completed. This notebooks writes the file received.csv.
4. Exploratory_Data_Analysis.ipynb ist for data exploration. Takes as input the cleaned files and the received.csv from Labeling
5. ML_preprocessing.ipynb performs some additional preprocessing steps to prepare the data for machine learning algorithms
6. xgboost_completed.ipynb is the notebook which creates the xgboost machine learning model
7. LinearLearner_completed.ipynb is the notebook  which creates the Linear Learner machine learning model

There are several more notebooks where I played with different features space, different input, pytorch etc. But these are not part of the final report.
