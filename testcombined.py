
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy
import numpy as np
import matplotlib
from keras.models import model_from_yaml
from keras.layers import Flatten
from keras.callbacks import EarlyStopping

from keras.models import model_from_json

matplotlib.matplotlib_fname()




def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(np.hstack(y_true),dtype=float), np.array(y_pred,dtype=float)
    print(y_true.shape,y_pred.shape)
    y_diff = y_true - y_pred
    dr = np.divide((y_diff) , y_true,out = np.zeros_like(y_diff),where=y_true != 0) 
    mae = (np.absolute(y_true - y_pred))
    print(dr.shape)
    tmp = (np.abs(dr)) * 100

    return tmp,mae




# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg


#master file , predicts t+1 for a given file .
dataset = read_csv('/Users/charan/minutelyMacdata_test.csv', header=0, index_col=0)

cols=[ 'headcount_unique','total_wait_time', 'month', 'day', 'dow', 'hour', 'min','device1','device2','device3','device4','device5','device6','device7','device8']
dataset=dataset[cols]
#print(dataset.columns.tolist())
#print(type(dataset))
values = dataset.values
#print(type(values))
#print(values[0:1,])
encoder = LabelEncoder()
values[:,2] = encoder.fit_transform(values[:,2])
values[:,3] = encoder.fit_transform(values[:,3])
values[:,4] = encoder.fit_transform(values[:,4])
values[:,5] = encoder.fit_transform(values[:,5])
values[:,6] = encoder.fit_transform(values[:,6])


values = values.astype('float32')
print(values[0:2,])
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
print(scaled[0:2,:])

maes=[]
rmses=[]


n_hours = 3  # to t-3
n_features = 15
reframed = series_to_supervised(scaled, n_hours, 1)
values = reframed.values

#print(reframed.columns.tolist())
print("reframe",reframed.shape,scaled.shape)
print(reframed.head())

n_train_hours = 30000 #so many rows  30000 for 1722 and 15950 for 2437
train = values[3:n_train_hours, :]
test = values[30000:, :] #rest  9979
# split into input and outputs
n_obs = n_hours * n_features


train_X, train_y = train[:, :n_obs], train[:, -n_features] #
test_X, test_y = test[:, :n_obs], test[:, -n_features]

print(train_X.shape, len(train_X), train_y.shape)
#print(train_X[0:1,:])
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
print(train_X.shape[0],train_X.shape[1],train_X.shape[2], train_y.shape, test_X.shape, test_y.shape)

 
# later...
 
# load YAML and create model
yaml_file = open('model_device_stateful_1_100.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = model_from_yaml(loaded_model_yaml)
# load weights into new model
loaded_model.load_weights("model_device_stateful_1_100.h5")
print("Loaded model from disk")




# make a prediction
yhat = loaded_model.predict(test_X,batch_size=1)
#print("shape  "+str(yhat.shape))
test_X = test_X.reshape((test_X.shape[0], n_hours*n_features))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, -14:]), axis=1)
#print('here1  '+str(inv_yhat.shape))
print(inv_yhat.shape)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
inv_yhat = [abs(x) for x in inv_yhat]
# inverinsert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, -14:]), axis=1)
#print('here '+inv_y.shape)
#print(" before y"+str(test_y[1:100]))

inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
#actual
#print( "actual vs predicted length ",len(inv_y),len(inv_yhat))
#print("  y"+str(inv_y[1:100]))

#predicted
#print("  yhat"+str(inv_yhat[1:100]))
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
#print('Test RMSE: %.3f' % rmse)
mape,mae = mean_absolute_percentage_error(inv_y, inv_yhat)
  
mape[mape == 0] = np.nan
rmses.append(rmse)
maes.append(np.nanmean(mae))

print('iteration :, Test RMSE:, Test MAPE:,Test MAE: %d %.3f %.3f %.3f' % (1,rmse,np.nanmean(mape),np.nanmean(mae)))

tmmape = list(mape.flatten())


import pandas as pd

combined = pd.DataFrame(
    {'actual': inv_y,
     'predicted': inv_yhat,
      'mape':pd.Series(tmmape),
      'mae' :pd.Series(mae.flatten())
    })

combined.to_csv("/users/charan/predicted_device_1722_1440_combined.csv", sep=',')

pyplot.plot(rmses,label='rmse')
pyplot.plot(maes,label='mae')
pyplot.xticks([1,2,3,4,5])
pyplot.xlabel("lags")


#pyplot.plot(inv_y, label='actual')
#pyplot.plot(inv_yhat, label='predicted')

pyplot.legend()
pyplot.show()

