from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.utils import check_array
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from matplotlib import pyplot
import numpy
import numpy as np
from datetime import datetime, timedelta


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(np.hstack(y_true),dtype=float), np.array(y_pred,dtype=float)
    print(y_true.shape,y_pred.shape)
    y_diff = y_true - y_pred
    dr = np.divide((y_diff) , y_true,out = np.zeros_like(y_diff),where=y_true != 0) 

    print(dr.shape)
    tmp = (np.abs(dr)) * 100

    return tmp


# date-time parsing function for loading the dataset
def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')
 
# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
	df = DataFrame(data)
	columns = [df.shift(i) for i in range(1, lag+1)]
	columns.append(df)
	df = concat(columns, axis=1)
	df.fillna(0, inplace=True)
	return df
 
# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)
 
# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]
 
# scale train and test data to [-1, 1]
def scale(train, test):
	# fit scaler
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaler = scaler.fit(train)
	# transform train
	train = train.reshape(train.shape[0], train.shape[1])
	train_scaled = scaler.transform(train)
	# transform test
	test = test.reshape(test.shape[0], test.shape[1])
	test_scaled = scaler.transform(test)
	return scaler, train_scaled, test_scaled
 
# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
	new_row = [x for x in X] + [value]
	array = numpy.array(new_row)
	array = array.reshape(1, len(array))
	inverted = scaler.inverse_transform(array)
	return inverted[0, -1]
 
# fit an LSTM network to training data
def fit_lstm(train, batch_size, nb_epoch, neurons):
	X, y = train[:, 0:-1], train[:, -1]
	X = X.reshape(X.shape[0], 1, X.shape[1])
	model = Sequential()
	model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	for i in range(nb_epoch):
		model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
		model.reset_states()
	return model
 
# make a one-step forecast
def forecast_lstm(model, batch_size, X):
	X = X.reshape(1, 1, len(X))
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0,0]
 
def dateparse (time_in_mins):    
	date = datetime(1970, 1, 1) + timedelta(seconds=int(time_in_mins)*60)
	return date
	#return datetime.datetime.fromtimestamp(float(time_in_mins*60))

# load dataset
series = read_csv('/Users/charan/minutelyMacdata.csv', header=0, parse_dates=True,date_parser=dateparse, index_col='epoch_min', squeeze=True)
cols=[ 'headcount_unique']
dataset=series[cols]
#print(dataset.columns.tolist())
#print(type(dataset))
raw_values = dataset.values
length = len(raw_values)
print(length)
testLen=round(.25*len(raw_values)) 
# transform data to be stationary
#raw_values = series.values
#diff_values = difference(raw_values, 1)
 
# transform data to be supervised learning
supervised = timeseries_to_supervised(raw_values, 1)
supervised_values = supervised.values
 
# split data into train and test-sets
train, test = supervised_values[0:-testLen], supervised_values[-testLen:]
 
# transform the scale of the data
scaler, train_scaled, test_scaled = scale(train, test)
 
print("fit the model",len(train_scaled))


lstm_model = fit_lstm(train_scaled, 1, 3000, 4)
# forecast the entire training dataset to build up state for forecasting
train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
lstm_model.predict(train_reshaped, batch_size=1)
 
# walk-forward validation on the test data
predictions = list()
for i in range(len(test_scaled)):
	# make one-step forecast
	X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
	yhat = forecast_lstm(lstm_model, 1, X)
	# invert scaling
	yhat = invert_scale(scaler, X, yhat)
	# invert differencing
	#yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
	# store forecast
	predictions.append(yhat)
	expected = raw_values[len(train) + i ]
	#print('Min=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))
 
# report performance
rmse = sqrt(mean_squared_error(raw_values[-testLen:], predictions))
mape = mean_absolute_percentage_error(raw_values[-testLen:], predictions)

mape[mape == 0] = np.nan

print('iteration :, Test RMSE:, Test MAPE: %d %.3f %.3f' % (j ,rmse,np.nanmean(mape)))

# line plot of observed vs predicted
pyplot.plot(raw_values[-testLen:])
pyplot.plot(predictions)
print(len(list(mape.flatten())))
print(raw_values.shape)
tmmape = list(mape.flatten())


import pandas as pd

combined = pd.DataFrame(
    {'actual': pd.Series(raw_values[-testLen:,0]),
     'predicted': pd.Series(predictions),
     'mape':pd.Series(tmmape)
    })





combined.to_csv("/users/charan/univariate_2437.csv", sep=',')


#pyplot.show()