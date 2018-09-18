# without devices , function of wait time and time period only
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import numpy
import numpy as np
import matplotlib
from keras.models import model_from_yaml

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

# convert series to  learning
def convert_to_time_independent(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(n_out, n_out+1):
		cols.append(df.shift(-i))
		print(i)
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



dataset = read_csv('/Users/charan/minutelyMacdata.csv', header=0, index_col=0)

cols=[ 'headcount_unique','total_wait_time', 'month', 'day', 'dow', 'hour', 'min']

dataset=dataset[cols]
values = dataset.values
encoder = LabelEncoder()
values[:,2] = encoder.fit_transform(values[:,2])
values[:,3] = encoder.fit_transform(values[:,3])
values[:,4] = encoder.fit_transform(values[:,4])
values[:,5] = encoder.fit_transform(values[:,5])
values[:,6] = encoder.fit_transform(values[:,6])



future=12 #predict t+12
testLen = 174
n_mins = 150  #lag of 150
  # to t-3
n_features = 7


values = values.astype('float32')
print(len(values))
data=values
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
flag = False


predicted=data[-(testLen-future-n_mins):,0]
scaled = scaler.fit_transform(data)
scaled=scaled[-testLen	:,]
print(len(scaled))


	
values = scaled
print(len(values))
#print(values)




reframed = convert_to_time_independent(values, n_mins, future)
print(reframed.shape)
print(reframed.columns)

reframed.to_csv("/users/charan/reframed.csv", sep=',')


values = reframed.values
test = values
n_obs = n_mins * n_features
test_X, test_y = test[:, :n_obs], test[:, -n_features]
test_X = test_X.reshape((test_X.shape[0], n_mins, n_features))
print(test_y)

# later...
 
# load YAML and create model
yaml_file = open('model.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = model_from_yaml(loaded_model_yaml)
# load weights into new model
loaded_model.load_weights("model.h5")
#print("Loaded model from disk")






# make a prediction
yhat = loaded_model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], n_mins*n_features))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, -6:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
inv_yhat = [abs(x) for x in inv_yhat]

test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, -6:]), axis=1)
#print('here '+inv_y.shape)
#print(" before y"+str(test_y[1:100]))

inv_y = scaler.inverse_transform(inv_y)
print(inv_y)
inv_y = inv_y[:,0]
#
print(predicted,inv_yhat)

rmses = sqrt(mean_squared_error(predicted, inv_yhat))
#print('Test RMSE: %.3f' % rmse)
mape,mae = mean_absolute_percentage_error(predicted, inv_yhat)
  
mape[mape == 0] = np.nan


print('iteration :, Test RMSE:, Test MAPE:,Test MAE: %d %.3f %.3f %.3f' % (1,np.mean(rmses),np.nanmean(mape),np.nanmean(mae)))

tmmape = list(mape.flatten())


import pandas as pd

combined = pd.DataFrame(
    {'actual': inv_y,
     'predicted': inv_yhat,
     'rmse':pd.Series(rmses),
      'mape':pd.Series(tmmape),
      'mae' :pd.Series(mae.flatten())
    })

combined.to_csv("/users/charan/predicted_1722.csv", sep=',')
#print('iteration :, Test RMSE:, Test MAPE:,Test MAE: %d %.3f %.3f %.3f' % (1,rmse,mape,mae))
