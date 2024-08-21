import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_squared_error, r2_score

def scale_dataset(X_train, X_test, y_train, y_test):
  X_train = X_train.apply(lambda x: x/100)
  X_test = X_test.apply(lambda x: x/100)
  y_train = y_train.apply(lambda x: x/100)
  y_test = y_test.apply(lambda x: x/100)

  return X_train, X_test, y_train, y_test

def grid_search_optimization(technique, X_train, y_train, seed):
  """
  Search for best hyperparams in search space for RF, DT, SVM and MLP algorithms
  Returns the best hyperparams for each algorithm (RF, DT, SVM and MLP)
  """
  if technique == "svm":
  
    #Tuning hyperparameters of a SVM-based water demand forecasting system through parallel global optimization
    svm_hyperparameters = {'kernel': ('linear', 'poly', 'rbf', 'sigmoid'), # Default = rbf
                          'C': [value for value in np.arange(1.0, 11, 1)], # Default = 1.0
                          #'degree': [value for value in range(2, 6)], # Default = 3
                          'gamma': ["scale", 0.0001, 0.001, 0.01, 0.1] # Default = scale
                          #'coef0': [value for value in np.arange(0.0, 1.1, 0.1)] # Default = 0.0
                          }


    svr = svm.SVR()
    
    optimized_svm = GridSearchCV(svr, svm_hyperparameters, n_jobs=-1, refit=True, error_score='raise', cv=TimeSeriesSplit(n_splits=5))

    optimized_svm.fit(X_train, y_train)
    #print(optimized_svm.best_estimator_)

    return optimized_svm

  elif technique == "rf":

    rf_hyperparameters = {'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'], # Default = squared_error
                          'n_estimators': [10, 50, 100], # Default = 100
                          'max_features': ["sqrt", "log2", None], # Default = None
                          'max_depth':  [50, None], # Default = None
                          #'min_samples_split':  [2, 11], # Default = 2
                          #'min_samples_leaf': [1, 11], # Default = 1
                          #'bootstrap': [True, False]
                          'random_state': [seed]
                        }
    
    rf = RandomForestRegressor(random_state=seed)
  
    optimized_rf = GridSearchCV(rf, rf_hyperparameters, n_jobs=-1, refit=True, error_score='raise', cv=TimeSeriesSplit(n_splits=5))

    optimized_rf.fit(X_train, y_train)
    #print(optimized_rf.best_estimator_)

    return optimized_rf
  
  elif technique == 'dt':

    dt_hyperparameters = {'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'], # Default = squared_error
                            'max_features': ["sqrt", "log2", None], # Default = None
                            'max_depth':  [50, None], # Default = None
                            #'min_samples_leaf': [1, 2, 3, 5], # Default = 1
                            #'min_samples_split':  [2, 4, 6, 8] # Default = 2
                            #'splitter': ["best", "random"] # Default = best
                            'random_state': [seed]
                          } 
  
    dt = DecisionTreeRegressor(random_state=seed)
  
    optimized_dt = GridSearchCV(dt, dt_hyperparameters, n_jobs=-1, refit=True, error_score='raise', cv=TimeSeriesSplit(n_splits=5))

    optimized_dt.fit(X_train, y_train)
    #print(optimized_dt.best_estimator_)

    return optimized_dt
  
  elif technique == 'mlp':

    # Hyperparameter optimization for interfacial bond strength prediction between fiber-reinforced polymer and concrete
    mlp_hyperparameters = {'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100)], # Default = 100
                           'activation': ["logistic", "relu", "tanh"], # Default = relu
                           #'solver': ["lbfgs", "sgd", "adam"], # Default = adam
                           #'alpha': [0.00001, 0.0001] # Default = 0.0001
                           'random_state': [seed]
                          }

    mlp = MLPRegressor(random_state=seed)

    optimized_mlp = GridSearchCV(mlp, mlp_hyperparameters, n_jobs=-1, refit=True, error_score='raise', cv=TimeSeriesSplit(n_splits=5))

    optimized_mlp.fit(X_train, y_train)
    #print(optimized_mlp.best_estimator_)

    return optimized_mlp

  return None

def ml_prediction(technique, X_train, X_test, y_train, y_test):

  """
  Predicts the WQI for 30 times, each with a different seed from 0 to 29
  The result is the average of R^2 scores
  """

  targets = y_test

  rmse_avg = [0] * 30
  mse_avg = [0] * 30
  mae_avg = [0] * 30
  r2_avg = [0] * 30
  r_score_avg = [0] * 30

  for seed in range(1, 31):

    # Setting same numpy.random seed as random_state seed
    np.random.seed(seed)
    
    if technique == "rf":
      optimized_rf = grid_search_optimization("rf", X_train, y_train, seed)
      model = RandomForestRegressor(**optimized_rf.best_params_)

    elif technique == "dt":
      optimized_dt = grid_search_optimization("dt", X_train, y_train, seed)
      model = DecisionTreeRegressor(**optimized_dt.best_params_)

    elif technique == "mlp":
      optimized_mlp = grid_search_optimization("mlp", X_train, y_train, seed)
      model = MLPRegressor(**optimized_mlp.best_params_)
      
    elif technique == "svm":
      optimized_svm = grid_search_optimization("svm", X_train, y_train, seed)
      model = svm.SVR(**optimized_svm.best_params_)

    model.fit(X_train, y_train)
    pred_WQI = model.predict(X_test) # predicted WQI

    rmse_avg[seed-1] = root_mean_squared_error(targets, pred_WQI)
    mse_avg[seed-1] = mean_squared_error(targets, pred_WQI)
    mae_avg[seed-1] = mean_absolute_error(targets, pred_WQI)
    r2_avg[seed-1] = r2_score(targets, pred_WQI)
    r_score_avg[seed-1] = np.sqrt(r2_score(targets, pred_WQI))

  return model, pred_WQI, rmse_avg, mse_avg, mae_avg, r2_avg, r_score_avg  # instance, array, array, array, ...

def store_metrics(row_index, svm_metrics_list, rf_metrics_list, dt_metrics_list, mlp_metrics_list, svm_metrics_values, rf_metrics_values, dt_metrics_values, mlp_metrics_values):
  """
  Store Evaluation Metrics results
  Receives arrays for metrics results for each ML algorithm
  Returns an array for each evaluation metrics with the respective results
  """

  svm_metrics_list.append({"RMSE": svm_metrics_values[0], "MSE": svm_metrics_values[1], "MAE": svm_metrics_values[2], "R2": svm_metrics_values[3], "R": svm_metrics_values[4]})

  rf_metrics_list.append({"RMSE": rf_metrics_values[0], "MSE": rf_metrics_values[1], "MAE": rf_metrics_values[2], "R2": rf_metrics_values[3], "R": rf_metrics_values[4]})

  dt_metrics_list.append({"RMSE": dt_metrics_values[0], "MSE": dt_metrics_values[1], "MAE": dt_metrics_values[2], "R2": dt_metrics_values[3], "R": dt_metrics_values[4]})

  mlp_metrics_list.append({"RMSE": mlp_metrics_values[0], "MSE": mlp_metrics_values[1], "MAE": mlp_metrics_values[2], "R2": mlp_metrics_values[3], "R": mlp_metrics_values[4]})                              

  return svm_metrics_list, rf_metrics_list, dt_metrics_list, mlp_metrics_list