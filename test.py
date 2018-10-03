import numpy as np
import pickle
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# TEST DATA PREDICTION - Part 3
#
# Two one week sets are used in this section for prediction.
# One contains the actual usage values (labeled) and the other does not.

# Importing XGBoost model and labeled test dataset
regressor = pickle.load(open("xgboost.dat", "rb"))
data = np.load('data_xgboost.npz')
test_data = data['arr_3']
X_test_labeled = test_data[:, 1:]
y_test = test_data[:, 0].reshape(-1, 1)

# Labeled test data prediction - It assumes that after each prediction measurements indicating the actual values
# of variables is available
training_data = data['arr_2']
id_arr = data['arr_4']
len_household = int(training_data.shape[0] / len(id_arr))
len_household_test = int(y_test.shape[0] / len(id_arr))
y_pred_test = []
for i in range(len(id_arr)):
    X_last_train = training_data[(i + 1) * len_household - 1, 1:].reshape(-1, 1)
    y_last_train = training_data[(i + 1) * len_household - 1, 0]
    new_x = np.array([y_last_train, y_last_train - X_last_train[0][0]]).reshape(-1, 1)
    X_test = np.concatenate((new_x, X_last_train[:-2]))
    X_test = np.reshape(X_test, (1, X_test.shape[0]))
    for j in range(len_household_test):
        y_regressor = regressor.predict(X_test)
        y_pred_test.append(y_regressor[0])
        y_real = y_test[i * len_household_test + j]
        new_x = np.reshape(np.array([y_real, y_real - X_test[0][0]]), (1, 2))
        X_test = np.concatenate((new_x, X_test[:, :-2]), axis=1)
y_pred_test = np.array(y_pred_test).reshape(-1, 1)
mse_test = mean_squared_error(y_test, y_pred_test)

for i in range(len(id_arr)):
    plt.plot(y_test[len_household_test*i:len_household_test*(i+1), 0], color='red', label=[id_arr[i] + ' actual values'])
    plt.plot(y_pred_test[len_household_test*i:len_household_test*(i+1), 0], color='blue', label=[id_arr[i] +
             ' predicted values'])
    plt.title('Usage for the ' + id_arr[i] + ' household')
    plt.xlabel('Time')
    plt.ylabel('Usage')
    plt.legend()
    plt.show()
