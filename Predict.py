#predicting one value iteratively 
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
def convert_to_time_independent(data, n_in=1, n_out=1, dropnan=True):
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



def time_series_to_supervised(data, n_lag=1, n_fut=1, selLag=None, selFut=None, dropnan=True):
    """
    Converts a time series to a supervised learning data set by adding time-shifted prior and future period
    data as input or output (i.e., target result) columns for each period
    :param data:  a series of periodic attributes as a list or NumPy array
    :param n_lag: number of PRIOR periods to lag as input (X); generates: Xa(t-1), Xa(t-2); min= 0 --> nothing lagged
    :param n_fut: number of FUTURE periods to add as target output (y); generates Yout(t+1); min= 0 --> no future periods
    :param selLag:  only copy these specific PRIOR period attributes; default= None; EX: ['Xa', 'Xb' ]
    :param selFut:  only copy these specific FUTURE period attributes; default= None; EX: ['rslt', 'xx']
    :param dropnan: True= drop rows with NaN values; default= True
    :return: a Pandas DataFrame of time series data organized for supervised learning
    NOTES:
    (1) The current period's data is always included in the output.
    (2) A suffix is added to the original column names to indicate a relative time reference: e.g., (t) is the current
        period; (t-2) is from two periods in the past; (t+1) is from the next period
    (3) This is an extension of Jason Brownlee's series_to_supervised() function, customized for MFI use
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    origNames = df.columns
    cols, names = list(), list()
    # include all current period attributes
    cols.append(df.shift(0))
    names += [('%s' % origNames[j]) for j in range(n_vars)]

    # lag any past period attributes (t-n_lag,...,t-1)
    n_lag = max(0, n_lag)  # force valid number of lag periods
    for i in range(n_lag, 0, -1):
        suffix= '(t-%d)' % i
        if (None == selLag):   # copy all attributes from PRIOR periods?
            cols.append(df.shift(i))
            names += [('%s%s' % (origNames[j], suffix)) for j in range(n_vars)]
        else:
            for var in (selLag):
                cols.append(df[var].shift(i))
                names+= [('%s%s' % (var, suffix))]

    # include future period attributes (t+1,...,t+n_fut)
    n_fut = max(n_fut, 0)  # force valid number of future periods to shift back
    for i in range(1, n_fut + 1):
        suffix= '(t+%d)' % i
        if (None == selFut):  # copy all attributes from future periods?
            cols.append(df.shift(-i))
            names += [('%s%s' % (origNames[j], suffix)) for j in range(n_vars)]
        else:  # copy only selected future attributes
            for var in (selFut):
                cols.append(df[var].shift(-i))
                names += [('%s%s' % (var, suffix))]
    # combine everything
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values introduced by lagging
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




values = values.astype('float32')
print(len(values))
data=values
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
predicted = 0
flag = False
n_hours = 3 # to t-3
n_features = 7
scaled = scaler.fit_transform(data)
scaled=scaled[-14:,]
print(len(scaled))
print(data[-10:,0])

for i in range(len(scaled)-n_hours-1):
		
	values = scaled[0:i+n_hours+1,]
	print("values",i,len(values))
	print("before",values[n_hours+i,0],values[n_hours-1+i,0])
	values[n_hours+i,0]=values[n_hours-1+i,0]
	#print(values)
	DataFrame(values).to_csv("/users/charan/reframed"+str(i)+".csv", sep=',')

	reframed = convert_to_time_independent(values, n_hours, 1)
	print(reframed.shape)
	print(reframed.head)
	values = reframed.values
	test = values
	n_obs = n_hours * n_features
	test_X, test_y = test[:, :n_obs], test[:, -n_features]
	test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))

	# later...


	# load YAML and create model
	yaml_file = open('model_1.yaml', 'r')
	loaded_model_yaml = yaml_file.read()
	yaml_file.close()
	loaded_model = model_from_yaml(loaded_model_yaml)
	# load weights into new model
	loaded_model.load_weights("model_1.h5")
	#print("Loaded model from disk")






	# make a prediction
	print(test_X.shape)
	yhat = loaded_model.predict(test_X)
	test_X = test_X.reshape((test_X.shape[0], n_hours*n_features))
	# invert scaling for forecast
	inv_yhat = concatenate((yhat, test_X[:, -6:]), axis=1)
	print("compare scaled",scaled[i+n_hours,0],inv_yhat[len(inv_yhat)-1:,0])
	scaled[i+n_hours,0]=inv_yhat[len(inv_yhat)-1:,0]
	inv_yhat = scaler.inverse_transform(inv_yhat)
	inv_yhat = inv_yhat[:,0]
	inv_yhat = [abs(x) for x in inv_yhat]


	print(inv_yhat[len(inv_yhat)-1])
	predicted = inv_yhat[len(inv_yhat)-1]
	print("comparing actual ",data[-(13-(i+n_hours)),0],predicted)	
	data[-(13-(i+n_hours)),0]=predicted


print(data[-10:,0])

