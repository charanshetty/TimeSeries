
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



dataset = read_csv('truncatedminutelyMacdata.csv', header=0, index_col=0)

cols=[ ' total_wait_time','headcount_unique', ' total_dwell_time', 'd1', 'd2', 'd3','month', 'day', 'dow', 'hour', 'min']
dataset=dataset[cols]
#print(dataset.columns.tolist())
#print(type(dataset))
values = dataset.values
#print(type(values))
#print(values[0:1,])
encoder = LabelEncoder()
values[:,6] = encoder.fit_transform(values[:,6])
values[:,7] = encoder.fit_transform(values[:,7])
values[:,8] = encoder.fit_transform(values[:,8])
values[:,9] = encoder.fit_transform(values[:,9])
values[:,10] = encoder.fit_transform(values[:,10])


values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
#print(scaled[1,:])
n_hours = 3
n_features = 11
reframed = series_to_supervised(scaled, n_hours, 1)
#print(reframed.columns.tolist())
print(reframed.shape)



values = reframed.values
n_train_hours = 365 * 24
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
# split into input and outputs
n_obs = n_hours * n_features
print(n_obs)
train_X, train_y = train[:, :n_obs], train[:, -n_features]
test_X, test_y = test[:, :n_obs], test[:, -n_features]
print(train_X.shape, len(train_X), train_y.shape)
#print(train_X[0:1,:])
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
#pyplot.plot(history.history['loss'], label='train')
#pyplot.plot(history.history['val_loss'], label='test')
#pyplot.legend()
#pyplot.show()



# make a prediction
yhat = model.predict(test_X)
print("shape  "+str(yhat.shape))
test_X = test_X.reshape((test_X.shape[0], n_hours*n_features))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, -10:]), axis=1)
#print('here1  '+str(inv_yhat.shape))
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
print(test_y.shape)
inv_y = concatenate((test_y, test_X[:, -10:]), axis=1)
#print('here '+inv_y.shape)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
print("  y"+str(inv_y[1:100]))
print("  yhat"+str(inv_yhat[1:100]))
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)
pyplot.plot(inv_y, label='actual')
pyplot.plot(inv_yhat, label='predicted')
pyplot.legend()
pyplot.show()