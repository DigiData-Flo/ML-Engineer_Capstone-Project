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
 
There are several noteboos for better organization.
