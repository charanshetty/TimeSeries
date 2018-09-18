# decomposes the file to residual,seasonality ans trend. 
#issue : how to reuse it in the test data .
from pandas import Series
from matplotlib import pyplot
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas import read_csv,DataFrame
from datetime import datetime, timedelta



#series = Series.from_csv('airline-passengers.csv', header=0,sep=";")


def dateparse (time_in_mins):    
	date = datetime(1970, 1, 1) + timedelta(seconds=int(time_in_mins)*60)
	return date
	#return datetime.datetime.fromtimestamp(float(time_in_mins*60))

 
def difference(dataset, interval=1440):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return diff
 
# invert differenced forecast
def inverse_difference(last_ob, value):
	return value + last_ob



series = read_csv('/Users/charan/minutelyMacdata_1722.csv', header=0, parse_dates=True, index_col='epoch_min', squeeze=True)
cols=[ 'headcount_unique','total_wait_time', 'month', 'day', 'dow', 'hour', 'min','device1','device2','device3','device4','device5','device6','device7','device8']
dataset=series[cols]
dataset.sort_index(ascending=False)
min = min(dataset.index)
max = max(dataset.index)

dataset=dataset.copy()
print(max,min)
for i in dataset.index:
	if i-1440 in dataset.index:
		tmp = dataset.loc[i,'headcount_unique']-dataset.loc[i-1440,'headcount_unique']
		dataset.loc[i,'headcount_unique']=tmp


dataset.to_csv("/users/charan/differenced.csv", sep=',')


#		dataset.iloc[i,'headcount']=dataset.iloc[i-1440,'headcount']

#diff = difference(dataset.values[1440:,], 1440)


#result_day = seasonal_decompose(DataFrame(diff), model='additive',freq=1440)
#result = seasonal_decompose(result_day.resid, model='additive',freq=10080)
#print(result_day.seasonal)

#(result_day.seasonal).plot()
#pyplot.show()




