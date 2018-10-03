# Questions
The dataset is this data challenge shows usage for 10 households over one year. Training data spans through 11 months and three weeks. Test data refers to usage prediction in the last week. Only training data shows usage values.

The following was considered to solve this open machine learning problem:

-	Features were defined as previous usage (depending on time delay) and usage difference between time steps (also depending on time delay). Targets were defined as current usage (1 target). This approach feeds historical data available from all households to a model in a sequential fashion (one household at a time). 

-	This seemingly simplistic single-input-single-output variable regression problem was setup so that each usage pattern in a given household influences future usage for all households whilst considering scalability, incorporation of new users and variable historical data for each user. Firstly, from a scalability perspective, this strategy would allow quick retraining of the network once new users are added to the database. Secondly, once new users’ data is available, it can be easily incorporated to the network and even used for updating the current weights. Finally, different users with different histories can be fed through the network and predicted once enough data is deemed enough. 

-	Usage prediction for each household is obtained by feeding only historical data relevant to that household after training. Two test data sets were conceived. The one provided by the data challenge itself represents the last week of measurements, where there is no usage data available. The other one was created by extracting the training data’s last week, where there is usage data available for reference. The test data given by this challenge considers that there are no new measurements to rely on when predicting new usage from now till one week into the future. By using not only this scenario, but also a scenario where new measurements are coming every 30 minutes, the model’s performance can be assessed more thoroughly.

## Your solution
### 1. What machine learning techniques did you use for your solution? Why?

XGBoost (Extreme Gradient Boosting) was considered for this challenge. XGBoost has reportedly shown faster training, and more accurate results then Random Forest and LSTM RNN. Despite not considering time delay in such an elegant way as LSTM, XGBoost can incorporate that by relying on time delayed features. Furthermore, as far as interpretability goes, LSTM suffers when trying to assess which features are more relevant for prediction, where XGBoost can provide certain measures that can help with this assessment, such as information gain estimates per feature.  

Time delay was set on 48 30-minute intervals, representing one day on, upon data inspection and exploration, roughly two usage cycles. This time delay was considered to be enough to capture the impact of time dependent features on the output.

XGBoost’s hyperparameters were tuned using grid search cross-validation on key elements relevant for predictive performance: max_depth, n_estimators, and learning_rate. XGBoost does not require feature scaling.

### 2. What is the error of your prediction?

The error of the prediction can be assessed in two ways, for the labelled test data (LTD) set and for the unlabelled (UTD) one.

   #### a. How did you estimate the error?
   
   LTD: Mean Square Error (MSE).
   
   UTD: How representative the prediction is when compared to the trend exhibited is the training dataset. 

   #### b. How do the train and test errors compare?
   
   LTD: Test errors are in the same order of magnitude as training errors, but larger. This is fair considering that the model was fit originally on the training dataset that range and scale do not change significantly between sets.

   UTD: The magnitude of spikes in usage are being under-predicted. The positioning of those peaks, however, fits well with the cycle 

   #### c. How will this error change for predicting a further week/month/year into the future?
   
   LTD: Predictive capabilities will inherently degrade the longer the model goes without retraining or updating. Compared to unlabelled predictions, since new measurements are available every thirty minutes, degradation is minimised. 

UTD: Predictions within one week show realistic variation between cycles, which fits better with actual usage cycles observed. For longer periods, however, cycles tend to reach a uniform, static pattern that does not represent real usage. 
   
### 3. What improvements to your approach would you pursue next? Why?

Given computational and time limitations, the models were trained on only two types of features, with single output predictions. The results show fair accuracy, but without great predictive power into the future. The following suggestions could aid prediction in the future.

-	Include date as a feature. Date and date related features might have a positive influence on the assessment of future usage features.

-	Look for other features that might influence usage behaviour. Meteorological data, age groups, geographical data, etc.

-	Engineer other time related features, so to capture other trends that might be hidden in data and might be valuable for prediction. Rolling window statistics, for example. 

-	Develop predictive model whose output is an entire week into the future (multiple-output) rather than the next step (single-output).

-	Time delay set to one week. Ensure known data impact in future predictions up to one week. 

-	Finer grid for cross-validation or use of different hyperparameter strategies for hyperparameter tuning (Bayesian hyperparameter optimisation, for example).

### 4. Will your approach work for a new household with little/no half-hourly data? How would you approach forecasting for a new household?

If the historical data is greater than the time delay specified, it could be fed through the network and used for prediction. In the scenarios where a brand-new household is integrated to this system, however, predictions could still be made considering time independent features (age, region, education, wealth). By assessing the similarity between a new household against other ones already integrated to the prediction network, it is possible to estimate temporarily new household behaviour as the average usage between n most similar households. 
