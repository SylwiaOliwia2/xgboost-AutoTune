# xgboost-AutoTune
Package allows for auto-tuninng `xgbooxt.XGBRegressor` parameters. Model usues GridSearchCV. Tested for Python 3.5.2.

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

## More detailed description
In progres.
