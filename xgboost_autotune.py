# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 12:46:53 2018

@author: sylwia Mielnicka

modified by Brian Pennington - July 2021

# TODO: 
    6. dla GBM- jak jest regularyzacja- upewnij się, że nie wywala wówczas kodu (bo nie ma żądnego parametru do optymalizacji)
"""
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import bisect
import numpy as np
import math
from copy import copy

   
### check if user provided initial values to perform Gridsearch
   # if yes- user's parameters would replace respective params domain_params_dicts
def set_initial_params(dictionary, param_name, domain_array):
    if param_name in dictionary and bool(dictionary[param_name]):
        params_array = dictionary[param_name]
    else:
        params_array = domain_array
    return params_array 
    
    
    
# each value in dict passed to GridSearchCV must be array, like {'n_estimators': [100], 'learning_rate' : [0.01]}
# but the same value provided to model must be int/str: {'n_estimators': 100, 'learning_rate' : 0.01}
# the function below converts dict with arrays to dict with integers
def convert_dict_of_arrays(dictionary):
    ret = {}
    for key, value_arr in dictionary.items():
        ret[key] = value_arr[0]
    return ret 
    
    
    
# update parameter values by those returned by Gridsearch
def update_model_params(model, model_params, params_to_update):
    try:
        # print('update_model_params:')
        # print('Model params:  ', model_params)
        print('Updating params: ', params_to_update)
        if model.__class__.__name__ == 'Pipeline':
            prefix = model.steps[-1][0] + '__'
            name = model.steps[-1][0]
            model_params.update(params_to_update)
            # strip prefix when updating params
            update_params = { key[len(prefix):]: val for key, val in model_params.items()}
            # create new model and update params
            new_model = model.steps[-1][1].__class__(**update_params)
            # remove current model and append new model
            model.steps.pop(-1)
            model.steps.append([name, new_model])
        else:
            model_params.update(params_to_update)
            model = model.__class__(**model_params)
    except:
        # add messaging
        print("Cannot update model params. ")
        print('Model params:  ', model_params)
        print('Update params: ', params_to_update)
        pass
    return model
    
    
    
##  create new array of parameters, which will be further researched to find optimum param values 
def create_new_array(param_name, param_array, value, param_requirments, prefix):
    
    print('creating new array for {} using param_array, {} for value {}'.format(param_name, param_array, value))
    # param_key is param_name stripped of prefix
    param_key = param_name[len(prefix):]
    if value in param_array:
        position = param_array.index(value)
        if position == 0 :
            lowest_val = param_array[position]*2 - param_array[position+1]
            # check formal value requirements
            if lowest_val < param_requirments[param_key]['min']:
                lowest_val = param_requirments[param_key]['min']
            new_array = [lowest_val, param_array[position], ((param_array[position+1] + param_array[position])/2) ]
        # if optimal value was the last in array - take higher values
        elif param_array[-1] == param_array[position]:
            # this doesn't seem right; doubling the final position then adding position-1?!?!?
            # ORIGINAL CODE:  highest_val = param_array[position]*2 + param_array[position-1] 
            # it seems more reasonable to make it symmetric and add the difference of position and position-1 divided by 2
            highest_val = param_array[position] + (param_array[position] - param_array[position-1] )/2
            # check formal value requirements
            if highest_val > param_requirments[param_key]['max']:
                highest_val = param_requirments[param_key]['max']
            new_array = [((param_array[position-1] + param_array[position])/2), param_array[position], highest_val]
        else:
            new_array = [((param_array[position-1] + param_array[position])/2), param_array[position], ((param_array[position+1] + param_array[position])/2) ]
    else:
        # if the new param is not in the original param_arry, i.e. it is an interpolated value, use bisection to come up with new array
        position = bisect.bisect_left(param_array, value)
        print('Interpolating new array from position {}'.format(position))
        if position <= 1 :
            lowest_val = param_array[position]*2 - param_array[position+1]
            # check formal value requirements
            if lowest_val < param_requirments[param_key]['min']:
                lowest_val = param_requirments[param_key]['min']
            new_array = [(lowest_val + value)/2, value, (param_array[position+1] + value)/2]
        # if optimal value was the last in array - take higher values
        elif param_array[-1] <= value:
            # add an increment based on the last two values of the param array
            new_array = [value, value + (param_array[-1] - param_array[-2])]            
        else:
            # take the average of the value and the surrounding values
            new_array = [(param_array[position-1] + value)/2, value, (param_array[position]+value)/2]              
    
    # check data type requirements
    if param_requirments[param_key]['type'] == 'int':
        # should be opposite
        # new_array[0] = math.ceil(new_array[0])
        # new_array[-1] = math.floor(new_array[-1])
        new_array[0] = math.floor(new_array[0])
        new_array[-1] = math.ceil(new_array[-1])
        new_array[1:-1] = np.round(new_array[1:-1])
    # remove duplicates:
    new_array = np.unique(new_array)
    print('new array:  ', new_array)
    return new_array
    
    
    
### perform Gridsearch over parameters and return the best model
def find_best_params(model, parameters, X_train, y_train, min_loss, scoring, n_folds, initial_score =0, prefix='', orig_parameters = None):
    if orig_parameters==None:
        orig_parameters = parameters
    param_requirments = {'subsample': {'max': 1, 'min': 1/len(X_train), 'type': 'float'}, # minimal value is fraction for one row
                         'colsample_bytree': {'max': 1, 'min': 1/X_train.shape[1], 'type': 'float'}, #  # minimal value is fraction for one column
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
    clf = GridSearchCV(model, parameters, scoring=scoring, verbose=0, cv = n_folds, refit=True)
    clf.fit(X_train, y_train)    
    # perform further searching if metric loss is still significant
    if 'ProbaScorer' in str(scoring.__class__):
        print('using predict_proba')
        new_score = scoring._score_func(clf.predict_proba(X_train), y_train) # calculate new metric_value
    else:
        new_score = scoring._score_func(clf.predict(X_train), y_train) # calculate new metric_value
    print('find_best_params:new_score:  ', new_score)
    #  the if condition is wrong; doesn't take into account of sign of scoring
    # ORIGINAL CODE:  if new_score-initial_score > min_loss:
    if initial_score==0. or (scoring.__dict__['_sign'] == 1 and new_score - initial_score >= min_loss) or (scoring.__dict__['_sign'] == -1 and initial_score - new_score >= min_loss):
        new_param_dict = {}
        for param_name, param_array in parameters.items():
            if len(param_array)>1:
                position = param_array.index(clf.best_params_[param_name])
                # create new array of parameters for further research based on best_value's position in array
                # if optimal value was the lowest in array - take lower values     
                new_array = create_new_array(param_name, orig_parameters[param_name], clf.best_params_[param_name], param_requirments, prefix)
                # assign new array if it's different than the old one
                if (len(new_array) != len(param_array)) or (new_array != param_array).any():
                    new_param_dict[param_name] = list(new_array)
        if len(new_param_dict)>0:
            find_best_params(model, new_param_dict, X_train, y_train, min_loss, scoring, n_folds, initial_score = new_score, prefix=prefix, orig_parameters = orig_parameters)

    return (clf)
    
    
    
## main function- find optimal_parameters for given function
def fit_parameters(initial_model, initial_params_dict, X_train, y_train, min_loss, scoring, n_folds=5):
    ### initial check
    available_models = ['XGBRegressor', 'GradientBoostingRegressor', 'LGBMRegressor', 'Pipeline']
    assert (type(initial_params_dict) is dict)
    # if pipline, assume model is that last entry and check that
    assert initial_model.__class__.__name__ in available_models or (initial_model.__class__.__name__ == 'Pipeline' and initial_model[-1].__class__.__name__ in available_models), 'Invalid model: ' + initial_model.__class__.__name__
    prefix = ''
    model=initial_model
    if initial_model.__class__.__name__ == 'Pipeline':
        available_params = list(model[-1].get_params().keys())
        prefix = model.steps[-1][0] + '__'
    else:
        available_params = list(model.get_params().keys())
    # print('Available params:  ', available_params)
    # domain parameters, which will be used if no parameters provided by user
        # 1. n_estimators- should be quite low, in range [40-120] (should be fast to check many parameters, n_estimators will be fine-tuned later) 
             # if optimal is 20, you might want to try lowering the learning rate to 0.05 and re-run grid search    
             # learning rate-  0.05-0.2 it should work early
             # for LightGmax_depthBM n_estimators: must be infinite (like 9999999) and use early stopping to auto-tune (otherwise overfitting)     
        # 2. num leaves- too much will lead to overfitting    
             # min_samples_split: This should be ~0.5-1% of min_split_gaintotal values.
             # min_child_weight:  (sample size / 1000), nevfor p_name, p_array in params_dict.items():ertheless depends on dataset and loss
        # 3. min_samples_leaf : a small value because of imbalanced classes, make combinations with the top 5 values min_samples_split 
        # 4. max_features = ‘sqrt’ : Its a general thumb-rule to start with square root. 
        # others:param_pair = {'n_estimators': [final_params['n_estimators'] * n], 'learning_rate' : [final_params['learning_rate'] / n]}
             # is_unbalance: false (make your own weighting with scale_pos_weight) 
             # Scale_pos_weight is the ratio of number of negative class to the positive class. Suppose, the dataset has 90 observations of negative class 
             #      and 10 observations of positive class, then ideal value of scale_pos_Weight should be 9
  
    domain_params_dicts = [{'n_estimators': [30, 50, 70, 100, 150, 200, 300]},                                      
                            {'max_depth': [3, 5, 7, 9], 'min_child_weight': [0.001, 0.1, 1, 5, 10, 20], 'min_samples_split': [1,2,5,10,20,30], 'num_leaves': [15, 35, 50, 75, 100,150]},
                            {'gamma': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], 'min_samples_leaf': [1,2,5,10,20,30], 'min_child_samples': [2,7,15,25,45], 'min_split_gain': [0, 0.001, 0.1, 1,5, 20]},
                            {'n_estimators': [30, 50, 70, 100, 150, 200, 300],  'max_features': range(10,25,3)},
                            {'subsample': [i/10 for i in range(4,10)], 'colsample_bytree': [i/10 for i in range(4,10)], 'feature_fraction': [i/10 for i in range(4,10)]},
                            {'reg_alpha':[1e-5, 1e-2, 0.1, 1, 25, 100], 'reg_lambda':[1e-5, 1e-2, 0.1, 1, 25, 100]}]

    # iterate over parameter names from domain_params_dicts, and adjust parameter value from following dictionaries
    for params_dict in domain_params_dicts:
        params ={}
        for p_name, p_array in params_dict.items():
            if (p_name in available_params):            
                params[prefix + p_name] = set_initial_params(initial_params_dict, p_name, p_array)
            
        # save new best parameters
        best_params = find_best_params(model, params, X_train, y_train, min_loss, scoring, n_folds, prefix=prefix, orig_parameters=None).best_params_
        if initial_model.__class__.__name__ == 'Pipeline':
            final_params = copy(model[-1].get_params())
            final_params = {prefix + key: val for key, val in final_params.items()}
        else:
            final_params = copy(model.get_params())
        model = update_model_params(model, final_params, best_params)
        # print('Final params:  ', final_params) 
        #
        # This was outside the loop as a final optimization, but I put it inside with the idea that as we adjust the parameters, we may need to adjust the n_estimators/learning_rate
        # finally adjust pair (n_estimators, learning_rate)
    try:
        best_score = 0
        same_score = False
        prev_score = 0
        # 8 and 15 seemed to be overkill
        # for n in [.5, 1, 2, 4, 8, 15]:
        for n in [.5, 1, 2, 4]:
            key = prefix + 'n_estimators'
            if n==0:
                mult = .5
            else:
                mult = n
            # param_pair = {'n_estimators': [final_params['n_estimators'] * n], 'learning_rate' : [final_params['learning_rate'] / n]}
            # print('n_estimators: ', final_params[prefix+'n_estimators'])
            # print('learning_rate: ', final_params[prefix+'learning_rate'])
            param_pair = {prefix+'n_estimators': [int(final_params[prefix+'n_estimators'] * mult)], prefix+'learning_rate' : [final_params[prefix+'learning_rate'] / mult]}
            print('prediction for: ', param_pair)
            clf = GridSearchCV(model, param_pair, scoring=scoring, verbose=0, cv = n_folds, refit=True)
            clf.fit(X_train, y_train) 
            new_score = scoring._score_func(clf.predict(X_train), y_train) # calculate new metric_value
            # save parameters, if they give better results
            # this shouldn't be here!!!
            # ORIGINAL CODE:  best_param_pair = param_pair
            if best_score == 0:
                if new_score==0:
                    print('The score is 0.  It appears that something is wrong')
                    break
                else:
                    print('Setting best score to {}'.format(new_score))
                    best_score = new_score
                    best_param_pair = param_pair
            elif abs(new_score - prev_score)<1.e-10:
                print('scores are same: {}'.format(new_score))
                # break if score doesn't change in two consecutive rounds
                if same_score:
                    print('Score had not changed in two interations.  Stopping calculation.')
                    break
                same_score = True
            elif scoring.__dict__['_sign'] == 1: # for score where greater is better
                if new_score - best_score >= min_loss:
                    same_score = False
                    print('Updating best score, {}, with new score, {}, because the difference, {}, was greater than the min_loss, {}'.format(best_score, new_score, new_score - best_score, min_loss))
                    best_score = new_score
                    best_param_pair = param_pair
            elif scoring.__dict__['_sign'] == -1:# for score where lower is better
                # ORIGINAL CODE:  if new_score - best_score <= min_loss:
                #  above is wrong;  should be:
                if best_score - new_score >= min_loss:
                    same_score = False
                    print('Updating best score, {}, with new score, {}, because the difference, {}, was greater than the min_loss, {}'.format(best_score, new_score,best_score - new_score, min_loss))
                    best_score = new_score
                    best_param_pair = param_pair
            print('best score {}; best param pair {}'.format(best_score, best_param_pair))
            prev_score = new_score
        best_param_pair = convert_dict_of_arrays(best_param_pair)
        print('updating model with best param pair: ', best_param_pair)
        model = update_model_params(model, final_params, best_param_pair)
    except:
        # add messaging
        print(' **** Problem trying to adjust n_observation/learning_rate ****')
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

