import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from xgboost import XGBRegressor
import pickle
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error

def feature_engineering(dataset, time_delay=1):
    # Split data per household
    usage_household = dataset.pivot(index='datetime', columns='id', values='usage')
    usage_delayed = usage_household
    # Creating features per household
    for i in range(time_delay):
        # Previous usage as feature
        usage_iter = usage_household.shift(i+1)
        # Difference between usages as feature
        usage_diff_iter = usage_household.shift(i+1) - usage_household.shift(i+2)
        usage_delayed = usage_delayed.join(usage_iter, rsuffix=('_'+str(i+1)))
        usage_delayed = usage_delayed.join(usage_diff_iter, rsuffix=('_diff'+str(i+1)))
    usage_household = usage_delayed
    # Clearing nan values due to time delay
    usage_household = usage_household.iloc[1+time_delay:, :]

    # Extracting the last week as labeled test data for future comparison against
    # unlabeled test data
    usage_test_labeled = usage_household.iloc[-336:, :]
    usage_household = usage_household.iloc[:-336, :]

    # Combining household columns for both training and test dataset
    id_list = dataset['id'].unique().tolist()
    households = len(id_list)
    train_dataset = usage_household.iloc[:, 0]
    test_dataset = usage_test_labeled.iloc[:, 0]
    for i in range(households-1):  # Combining current usage values
        df = usage_household.iloc[:, i+1]
        df_test = usage_test_labeled.iloc[:, i+1]
        train_dataset = pd.concat([train_dataset, df])
        test_dataset = pd.concat([test_dataset, df_test])

    for i in range(2*time_delay):  # Combining features
        df_iter = usage_household.iloc[:, (i+1)*households]
        df_iter_test = usage_test_labeled.iloc[:, (i + 1) * households]
        for j in range(households-1):
            df = usage_household.iloc[:, (i+1)*households + j+1]
            df_iter = pd.concat([df_iter, df])
            df_test = usage_test_labeled.iloc[:, (i + 1) * households + j + 1]
            df_iter_test = pd.concat([df_iter_test, df_test])
        train_dataset = pd.concat([train_dataset, df_iter], axis=1)
        test_dataset = pd.concat([test_dataset, df_iter_test], axis=1)

    return usage_household, usage_test_labeled, train_dataset, test_dataset


# Importing the training dataset
dataset_train = pd.read_csv('usage_train.csv')

# Feature engineering with time delay = 48 (one day, roughly two usage cycles)
usage_household, usage_test_labeled, train_dataset, test_dataset = feature_engineering(dataset_train)
X_train = train_dataset.iloc[:, 1:].values
y_train = train_dataset.iloc[:, 0].values.reshape(-1, 1)

# BUILDING THE MODEL - Part 2

regressor = XGBRegressor()

# Applying grid search k-fold cross validation
parameters = {'max_depth': [3, 4, 5],
              'learning_rate': [0.2, 0.1, 0.01],
              'n_estimators': [50, 100, 200]}
grid_search = GridSearchCV(estimator=regressor,
                           param_grid=parameters,
                           scoring='neg_mean_squared_error',
                           cv=TimeSeriesSplit(n_splits=10),
                           verbose=10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_mse = grid_search.best_score_

regressor = XGBRegressor(n_estimators=best_parameters['n_estimators'], max_depth=best_parameters['max_depth'],
                         learning_rate=best_parameters['learning_rate'])
regressor.fit(X_train, y_train)

# Predicted values for training data
y_pred_train = regressor.predict(X_train).reshape(-1, 1)
# Mean squared error evaluation of training data
mse = mean_squared_error(y_train, y_pred_train)

# Displaying first 30 values for each household
id_list = dataset_train['id'].unique()
len_household = int(y_train.shape[0] / len(id_list))
for i in range(len(id_list)):
    plt.plot(y_train[len_household*i:200+len_household*i, 0], color='red', label=[id_list[i] + ' actual values'])
    plt.plot(y_pred_train[len_household*i:200+len_household*i, 0], color='blue', label=[id_list[i] + ' predicted values'])
    plt.title('Usage for the ' + id_list[i] + ' household')
    plt.xlabel('Time')
    plt.ylabel('Usage')
    plt.legend()
    plt.show()

# Saving the model as an h5 file
pickle.dump(regressor, open("xgboost.dat", "wb"))
# Saving formatted datasets
np.savez('data_xgboost', usage_household, usage_test_labeled, train_dataset, test_dataset, id_list)

