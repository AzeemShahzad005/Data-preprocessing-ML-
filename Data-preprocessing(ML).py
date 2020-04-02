# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 12:55:45 2020

@author: Azeem
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('D:\DS work\Machine Learning A-Z\Part 1 - Data Preprocessing\Data.csv')
 X = dataset.iloc[: , :-1].values
 Y = dataset.iloc[: , 3].values
 
 from sklearn.preprocessing import Imputer 
 imputer = Imputer(missing_values = 'NaN', strategy= 'mean', axis= 0)
 imputer = imputer.fit(X[: , 1:3])
 X[: , 1:3] = imputer.transform(X[: , 1:3])
 