import numpy as np
import pickle

# TEST DATA PREDICTION - Part 3
#
# Two one week sets are used in this section for prediction.
# One contains the actual usage values (labeled) and the other does not.

# Importing XGBoost model and labeled test dataset
regressor = pickle.load(open("xgboost.dat", "rb"))
data = np.load('data_xgboost.npz')
test_data = data['arr_3']
X_test_labeled = test_data[:, 1:-1]
y_test = test_data[:, 0].reshape(-1, 1)
