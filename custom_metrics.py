# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 13:24:35 2018

@author: sylwia
"""
from sklearn.metrics import make_scorer, accuracy_score
import numpy as np

def rmsle(h, y): 
    """
    Args:
        h - numpy array containing predictions with shape (n_samples, n_targets)
        y - numpy array containing targets with shape (n_samples, n_targets)
    """
    return np.sqrt(np.square(np.log(h + 1) - np.log(y + 1)).mean())   

rmlse_score = make_scorer(rmsle, greater_is_better=False)
accuracy = make_scorer(accuracy_score, greater_is_better=True)