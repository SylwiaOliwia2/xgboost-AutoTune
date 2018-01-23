# xgboost-AutoTune
Package allows for auto-tuninng `xgbooxt.XGBRegressor` parameters. Model usues GridSearchCV. Tested for Python 3.5.2.

## Preface
**Please keep in mind it's not tutorial on boosting methods**, but library for auto-tuning them. I assume you are familiar with bosting. If not, firstly visit the pages below: 

http://xgboost.readthedocs.io/en/latest/model.html 

https://machinelearningmastery.com/gentle-introduction-xgboost-applied-machine-learning/  

## Instalation
`pip3 install git+git://github.com/SylwiaOliwia2/xgboost-AutoTune@choose-params`

Note that xgboost-AutoTune depends on Numpy and Sklearn.

## Create scorer
Before runnheader_titleing you need to define scoring. The easier way is using `sklearn.metrics.make_scorer`. You can see example in [custom_metrics.py](https://github.com/SylwiaOliwia2/xgboost-AutoTune/blob/choose-params/custom_metrics.py).

## Fast run
To run it type in your code:

```
from xgboost_autotune import fit_parameters
from sklearn.metrics import make_scorer, accuracy_score

accuracy = make_scorer(accuracy_score, greater_is_better=True)
fitted_model = fit_parameters(initial_model = xgboost.XGBRegressor(), initial_params_dict = {}, X_train = X, y_train = y_train, min_loss = 0.01, scoring=accuracy, n_folds=5)
```

### Parameters:
* **initial_model** - `xgboost.XGBRegressor` model. You can leave parameters empty:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`model = xgboost.XGBRegressor()` or predefine them (they will be used, until better params will be chosen):

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`model = xgboost.XGBRegressor(max_depth = 7, learning_rate = 0.1,  colsample_bytree=0.7)`
* **initial_params_dict** - dictionary of values to be tested. If empty (as in the example)- domain parameters' arrays will be 
used for search. If you will overwrite domain parameters, provide arrays of values in the dictionary, ex:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`{'susample': [0.6, 0.8, 1], 'max_depth': [3,5,7]'}`.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;If you type only one value in parameter array then this particular value will be used, without tuning this parameter, ex:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`{'susample': [0.8], 'max_depth': [3,5,7]'}` will use `subsample` = 0.8 and wouldn't look for better `subsample` value.

* **X_train** - pandas DataFrame, input variables
* **y_train** - pandas series, labels for prediction
* **min_loss** - minimum scoring loss required for further parameter research (in neighbourhood of the best value).
* **scoring** - used to evalueate the best model. See **Create scorer** section above.
* **n_folds** - number of folds used in GridSearchCV

## Implementation details
### General note
Full GridSearch is time- and memory-demanding, so xgboost-AutoTune tunes parameters in the following steps (one by one, from the most robust to the less): 
1. n_estimators
2. max_depth, min_child_weight
3. Gamma                       
4. n_estimators
5. Subsample, colsample_bytree
6. reg_alpha, reg_lambda 
7. n_estimators and learning_rate 

Some of them are related only to `xgboost`, `LightGBM` or `GBM`. Algorithm picks parameters valid for given model and skip the rest. 

Model is updated by newly chosen parameters in each step. 

### Detailed notes
Algorithm make GridsearchCV for each in seven steps (see **General note** section) and choose the best value. It uses domian values: 
```
{'n_estimators': [30, 50, 70, 100, 150, 200, 300]},                                       
{'max_depth': [3, 5, 7, 9], 'min_child_weight': [0.001, 0.1, 1, 5, 10, 20], 'min_samples_split': [1,2,5,10,20,30], 'num_leaves': [15, 35, 50, 75, 100,150]}, 
{'gamma': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], 'min_samples_leaf': [1,2,5,10,20,30], 'min_child_samples': [2,7,15,25,45], 'min_split_gain': [0, 0.001, 0.1, 1,5, 20]}, 
{'n_estimators': [30, 50, 70, 100, 150, 200, 300],  'max_features': range(10,25,3)}, 
{'subsample': [i/10 for i in range(4,10)], 'colsample_bytree': [i/10 for i in range(4,10)], 'feature_fraction': [i/10 for i in range(4,10)]}, 
{'reg_alpha':[1e-5, 1e-2, 0.1, 1, 25, 100], 'reg_lambda':[1e-5, 1e-2, 0.1, 1, 25, 100]}
```
Unless user will provide his own dictionary of values in **initial_params_dict**. 

In each iteration, if chosing the best value from array has improved **scoring** by **min_loss**, algorithm continue searching. It creates new array from the best value, and 2 values in the neighbourhood: 

* If the best value in the previous array had neighbours, then new neighbours will be average between best value and it's previous neighbours. Example: if the best value from `n_estimators`: `[30, 50, 70, 100, 150, 200, 300]` will be 70, than the new array to search will be `[60, 70, 85]`. 

* If the best value is the lowest from the array, it's new value will be `2*best_value- following_value` unless it's bigger then minimal (otherwise minimal posible value). 

* The the best value was the biggest in the array, it will be treated in the similar way, as the lowest one. 

If new values are float and int is required, values are rounded. 

`n_estimators` and `learning_rate` are chosen pairwise. Algorithm takes its values from model and train them pairwise: (n* `n_estimators` , `learning_rate`/ n ). 

## Sources
xgboost-AutoTune bases on the advises from the posts below: 

http://xgboost.readthedocs.io/en/latest/how_to/param_tuning.html 

https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/ 

https://machinelearningmastery.com/tune-number-size-decision-trees-xgboost-python/ 

https://www.kaggle.com/prasunmishra/parameter-tuning-for-xgboost-sklearn/notebook 

https://cambridgespark.com/content/tutorials/hyperparameter-tuning-in-xgboost/index.html 
