# xgboost-AutoTune
Package allows for auto-tuninng xgbooxt (XGBRegressor) parameters. Model usues GridSearchCV. Tested for Python 3.5.2.
**Much more precise README and another GBM models comming soon**.

## Instalation
At this moment please download `Heuristic_choose_parameters.py` manually. 
**Installation via pip comming soon.**

## Create scorer
Before running you need to define scoring. The easier way is using `sklearn.metrics.make_scorer`. You can see example in `custom_metrics.py`.

## Run
To run it type in your code:
`from xgboost_autotune import fit_parameters`

`fitted_model = fit_parameters(xgb_model, {}, X_train, y_train, min_loss = 0.01, scoring=rmlse_score, n_folds=5)`

### Parameters:
* **xgb_model** - `xgboost.XGBRegressor` model
* **{}** - dictionary of values to be tested. If empty (as in the example)- domain parameters' arrays will be 
used for testing. If you will overwrite domain parameters, provide arrays of values in the dictionary, ex `{'susample: [0.8], max_depth: [3,5,7]'}`
* **min_loss**- minimum scoring loss required to look for more exact parameter values than in parameters arrays. 
Requires domain knowledge, should be adjusted to scoring.
* **scoring** - score loss, used to evalueate the best model. Cam be created by wrapping one of `sklearn.metrics` methods in `sklearn.metrics.make_scorer`.
* **n_folds** - number of folds used in GridSearchCV
