# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 12:46:53 2018

@author: sylwia Mielnicka

# TODO: 
    6. dla GBM- jak jest regularyzacja- upewnij się, że nie wywala wówczas kodu (bo nie ma żądnego parametru do optymalizacji)
"""
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import numpy as np
import math
from copy import copy

   
### check if user provided initial values to perform Gridsearch
   # if yes- users parameters would replace respecitve params domain_params_dicts
def set_initial_params(dictionary, param_name, domain_array):
    if param_name in dictionary and bool(dictionary[param_name]):
        params_array = dictionary[param_name]
    else:
        params_array = domain_array
    return params_array 
    
    
    
# each value in dict passed to GridSearchCV must be array, like {'n_estimators': [100], 'learning_rate' : [0.01]}
# but the same value provided to model must be int/str: {'n_estimators': 100, 'learning_rate' : 0.01}
# the function below converts dictwith arrays to dict with integers
def convert_dict_of_arrays(dictionary):
    ret = {}
    for key, value_arr in dictionary.items():
        ret[key] = value_arr[0]
    return ret 
    
    
    
# update parameter values by those returned by Gridsearch
def update_model_params(model, model_params, params_to_update):
    try:
        model_params.update(params_to_update)
        model = model.__class__(**model_params)
    except:
        pass
    return model
    
    
    
##  create new array of parameters, which will be further researched to find optimap param values 
def create_new_array(param_name, param_array, position, param_requirments):
    if position == 0 :
        lowest_val = param_array[position]*2 - param_array[position+1]
        # check formal value requirements
        if lowest_val < param_requirments[param_name]['min']:
            lowest_val = param_requirments[param_name]['min']
        new_array = [lowest_val, param_array[position], ((param_array[position+1] + param_array[position])/2) ]
    # if optimal value was the last in array - take higher values
    elif param_array[-1] == param_array[position]:
        highest_val = param_array[position]*2 + param_array[position-1] 
        # check formal value requirements
        if highest_val > param_requirments[param_name]['max']:
            highest_val = param_requirments[param_name]['max']
        new_array = [((param_array[position-1] + param_array[position])/2), param_array[position], highest_val]
    else:
        new_array = [((param_array[position-1] + param_array[position])/2), param_array[position], ((param_array[position+1] + param_array[position])/2) ]                    
    
    # check data type requirements
    if param_requirments[param_name]['type'] == 'int':
        new_array[0] = math.ceil(new_array[0])
        new_array[-1] = math.floor(new_array[-1])
        new_array[1:-1] = np.round(new_array[1:-1])
    # remove duplicates:
    new_array = np.unique(new_array)
    return new_array
    
    
    
### perform Gridsearch over parameters and return the best model
def find_best_params(model, parameters, X_train, y_train, min_loss, scoring, n_folds, iid, initial_socre =0):
    param_requirments = {'subsample': {'max': 1, 'min': 1/len(X_train), 'type': 'float'}, # minimal value is fraction for one row
                         'colsample_bytree': {'max': 1, 'min': 1/len(X_train.columns), 'type': 'float'}, #  # minimal value is fraction for one column
                         'reg_alpha': {'max': np.inf, 'min': 0, 'type': 'float'},
                         'reg_lambda': {'max': np.inf, 'min': 0, 'type': 'float'},
                         'reg_scale_pos_weightlambda': {'max': np.inf, 'min': 0, 'type': 'float'},
                         'learning_rate': {'max': 1, 'min': 1e-15, 'type': 'float'}, # technically it moght be more than 1, but it may lead to underfittting
                         'n_estimators': {'max': np.inf, 'min': 1, 'type': 'int'},
                         'max_features': {'max': np.inf, 'min': 1, 'type': 'int'},
                         'gamma': {'max': np.inf, 'min': 0, 'type': 'float'},
                         'min_samples_leaf':{'max': np.inf, 'min': 1, 'type': 'int'}, # could be float (then i'ts percentage of all examples, but we'll use integers (number of samples) for consistency)
                         'min_samples_split': {'max': np.inf, 'min': 1, 'type': 'int'}, # could be float (then i'ts percentage of all examples, but we'll use integers (number of samples) for consistency)
                         'min_child_samples':{'max': np.inf, 'min': 1, 'type': 'int'},
                         'min_split_gain': {'max': np.inf, 'min': 0, 'type': 'float'},
                         'min_child_weight': {'max': np.inf, 'min': 0, 'type': 'float'},
                         'max_depth': {'max': np.inf, 'min': 1, 'type': 'int'},
                         'num_leaves': {'max': np.inf, 'min': 1, 'type': 'int'}} 
                         
    assert (min_loss != 0) # if equal to 0 - would be calculated infinity
    print ("Find best parameters for: ", parameters)
    clf = GridSearchCV(model, parameters, scoring=scoring, verbose=0, cv = n_folds, refit=True, iid=iid)
    clf.fit(X_train, y_train)    
    
    # perform further searching if metric loss is still significant
    new_score = scoring._score_func(clf.predict(X_train), y_train) # calculate new metric_value
    if new_score-initial_socre > min_loss:
        new_param_dict = {}
        for param_name, param_array in parameters.items():
            if len(param_array)>1:
                position = param_array.index(clf.best_params_[param_name])
                # crete new array of parameters for further research based on best_value's position in array
                # if optimal value was the lowest in array - take lower values     
                new_array = create_new_array(param_name, param_array, position, param_requirments)
                # assign new array if it's different than the old one
                if (len(new_array) != len(param_array)) or (new_array != param_array).any():
                    new_param_dict[param_name] = list(new_array)
        if len(new_param_dict)>0:
            find_best_params(model, new_param_dict, X_train, y_train, min_loss, scoring, n_folds, iid, initial_socre = new_score)

    return (clf)
    
    
    
## main function- find optimal_parameters for given function
def fit_parameters(initial_model, initial_params_dict, X_train, y_train, min_loss, scoring, n_folds=5, iid=False):
    ### initial check
    available_models = ['XGBRegressor', 'GradientBoostingRegressor', 'LGBMRegressor']
    assert (type(initial_params_dict) is dict)
    assert (initial_model.__class__.__name__ in available_models)
    
    model=initial_model
    available_params = list(model.get_params().keys())
    # domain parameters, which will be used if no parameters provided by user
        # 1. n_estimators- should be quite low, in ranparamsge [40-120] (should be fast to checm many parameters, n_estimators will be fine-tuned later) 
             # if optimal is 20, you might want to try lowering the learning rate to 0.05 and re-run grid search    
             # learning rate-  0.05-0.2 powinno działać na początku 
             # for LightGmax_depthBM n_estimators: must be infinite (like 9999999) and use early stopping to auto-tune (otherwise overfitting)     
        # 2. num leaves- too much will lead to overfitting    
             # min_samples_split: This should be ~0.5-1% of min_split_gaintotal values.
             # min_child_weight:  (sample size / 1000), nevfor p_name, p_array in params_dict.items():ertheless depedns on dataset and loss
        # 3. min_samples_leaf : a small value because of imbalanced classes, zrób kombinacje z 5 najlepszymi wartościami min_samples_split 
        # 4. max_features = ‘sqrt’ : Its a general thumb-rule to start with square root. 
        # others:param_pair = {'n_estimators': [final_params['n_estimators'] * n], 'learning_rate' : [final_params['learning_rate'] / n]}
             # is_unbalance: false (make your own weighting with scale_pos_weight) 
             # Scale_pos_weight is the ratio of number of negative class to the positive class. Suppose, the dataset has 90 observations of negative class and 10 observations of positive class, then ideal value of scale_pos_Weight should be 9
  
    domain_params_dicts = [{'n_estimators': [30, 50, 70, 100, 150, 200, 300]},                                      
                            {'max_depth': [3, 5, 7, 9], 'min_child_weight': [0.001, 0.1, 1, 5, 10, 20], 'min_samples_split': [1,2,5,10,20,30], 'num_leaves': [15, 35, 50, 75, 100,150]},
                            {'gamma': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], 'min_samples_leaf': [1,2,5,10,20,30], 'min_child_samples': [2,7,15,25,45], 'min_split_gain': [0, 0.001, 0.1, 1,5, 20]},
                            {'n_estimators': [30, 50, 70, 100, 150, 200, 300],  'max_features': range(10,25,3)},
                            {'subsample': [i/10 for i in range(4,10)], 'colsample_bytree': [i/10 for i in range(4,10)], 'feature_fraction': [i/10 for i in range(4,10)]},
                            {'reg_alpha':[1e-5, 1e-2, 0.1, 1, 25, 100], 'reg_lambda':[1e-5, 1e-2, 0.1, 1, 25, 100]}]

    # iterate over parameter anmes from domain_params_dicts, and adjust parameter value from following dictionaries
    for params_dict in domain_params_dicts:
        params ={}
        for p_name, p_array in params_dict.items():
            if (p_name in available_params):            
                params[p_name] = set_initial_params(initial_params_dict, p_name, p_array)
            
        # save new best parameters
        best_params = find_best_params(model, params, X_train, y_train, min_loss, scoring, n_folds, iid).best_params_
        final_params = copy(model.get_params())
        model = update_model_params(model, final_params, best_params)
        
    # finally adjust pair (n_estimators, learning_rate)
    try:
        best_score = None
        for n in [1, 2, 4, 8, 15, 25]:
            param_pair = {'n_estimators': [final_params['n_estimators'] * n], 'learning_rate' : [final_params['learning_rate'] / n]}
            print('prediction for: ', param_pair)
            clf = GridSearchCV(model, param_pair, scoring=scoring, verbose=0, cv = n_folds, refit=True,  iid=iid)
            clf.fit(X_train, y_train) 
            new_score = scoring._score_func(clf.predict(X_train), y_train) # calculate new metric_value
            
            # save parameters, if they give better results
            best_param_pair = param_pair
            if best_score is None:
                best_score = new_score
            elif scoring.__dict__['_sign'] == 1: # for score where greater is better
                if new_score - best_score >= min_loss:
                    best_score = new_score
                    best_param_pair = param_pair
            elif scoring.__dict__['_sign'] == -1:# for score where lower is better
                if new_score - best_score <= min_loss:
                    best_score = new_score
                    best_param_pair = param_pair
            print ('best score', best_score)
        best_param_pair = convert_dict_of_arrays(best_param_pair)
        model = update_model_params(model, final_params, best_param_pair)
    except:
        pass
    model.fit(X_train, y_train) 

    return model


## TEST set_initial_params
assert (set_initial_params({'n_estimators': []}, 'n_estimators', [40,70,100]) == [40,70,100])
assert (set_initial_params({'ggg': []}, 'n_estimators', [40,70,100]) == [40,70,100])
assert (set_initial_params({'ggg': [2,3]}, 'n_estimators', [40,70,100]) == [40,70,100])
assert (set_initial_params({'n_estimators': [2,3]}, 'n_estimators', [40,70,100]) == [2,3])
assert (len( set_initial_params({'n_estimators': [30,50]}, 'n_estimators', [40,70,100])) > 0 )
assert (convert_dict_of_arrays({'a':[1], 'b':['something']}) == {'a':1, 'b':'something'})

