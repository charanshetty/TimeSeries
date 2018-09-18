
#calculating average and filtering based on the average. to get better average 
#calculate avg and std .. filter the data based on average and std 
#calculate new avg on filtering 

from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame,merge
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
from keras.layers import Flatten,TimeDistributed
from keras.callbacks import EarlyStopping

from keras.models import model_from_json


dataset = read_csv('/Users/charan/newavgs_2.csv', header=0, index_col='epoch_min')

cols=['headcount_unique','total_wait_time', 'month', 'day', 'dow', 'hour', 'min','device1','device2','device3','device4','device5','device6','device7','device8']
dataset=dataset[cols]

epochmod = dataset.index % 1440
dataset['epoch_min']=dataset.index

dataset['mod'] = epochmod
print(dataset.columns)
mean = dataset[['mod','headcount_unique']].groupby(['mod']).mean()
std =   dataset[['mod','headcount_unique']].groupby(['mod']).std()
mean['mod']=mean.index
print(mean.columns)
mean.columns=['avg_headcount','mod']
std['mod'] = std.index
std.columns=['std_headcount','mod']

print(mean.columns)
newdataset = merge(merge(dataset,mean,on='mod'),std,on='mod')
col1s=['headcount_unique','std_headcount','avg_headcount']
#print(newdataset[50:100][col1s])

tm = (newdataset.headcount_unique<newdataset.avg_headcount+3*newdataset.std_headcount)
print(tm.index)
newdataset.drop(newdataset[(newdataset.headcount_unique>newdataset.avg_headcount+1*newdataset.std_headcount) | (newdataset.headcount_unique<newdataset.avg_headcount-1*newdataset.std_headcount)].index, inplace=True)
print(newdataset.epoch_min)
print(newdataset.shape)
newdataset.to_csv('/Users/charan/newavgs_2.csv',sep=",")


