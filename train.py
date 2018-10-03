import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from xgboost import XGBRegressor
import pickle
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

def feature_engineering(dataset, time_delay=1):
    # Split data per household
    usage_household = dataset.pivot(index='datetime', columns='id', values='usage')
    usage_delayed = usage_household
    usage_diff = usage_household - usage_household.shift(1)
    # Creating features per household
    for i in range(time_delay):
        # Previous usage as feature
        usage_iter = usage_household.shift(i+1)
        # Difference between usages as feature
        usage_diff_iter = usage_household.shift(i+1) - usage_household.shift(i+2)
        usage_delayed = usage_delayed.join(usage_iter, rsuffix=('_'+str(i+1)))
        usage_delayed = usage_delayed.join(usage_diff_iter, rsuffix=('_diff'+str(i+1)))
    usage_household = usage_delayed
    usage_household = usage_household.join(usage_diff, rsuffix='diff')
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

    for i in range(2*time_delay + 1):  # Combining features
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
