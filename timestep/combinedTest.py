#main program which uses mixture of models to predict multistep in this case t+12
# models. t+1,t+2,t+3,t+4,t+5,t+6,t+7,t+8,t+9,t+10,t+11 again t+1 gave t+12 
#following is the mae for every timestep , goes bad with increase in timestep
#mae is independent of the value itself , while rmse a has a amplified effect on the differnce due to square
#t+1     t+2     t+3     t+4     t+5     t+6     t+7     t+8     t+9     t+10    t+11    t+12
#4.31    6.825   8.936   10.25   10.685  10.24   12.313  14.01   14.14   14.929  14.66   17.7
import os

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
from keras.layers import Dropout
from keras.layers import Activation, Dense,TimeDistributed


timeMap = {}
rmses=[]
maes=[]

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(np.hstack(y_true),dtype=float), np.array(y_pred,dtype=float)
    print(y_true.shape,y_pred.shape)
    y_diff = y_true - y_pred
    dr = np.divide((y_diff) , y_true,out = np.zeros_like(y_diff),where=y_true != 0) 
    mae = (np.absolute(y_true - y_pred))
    print(dr.shape)
    tmp = (np.abs(dr)) * 100

    return tmp,mae



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


dataset = read_csv('/Users/charan/minutelyMacdata_1722.csv', header=0, index_col=0)

cols=[ 'headcount_unique','total_wait_time', 'month', 'day', 'dow', 'hour', 'min','device1','device2','device3','device4','device5','device6','device7','device8']
dataset=dataset[cols]
#print(dataset.columns.tolist())
#print(type(dataset))

dataset=dataset.iloc[:30000,]

data_file = open("../../minutelyMacdata_1722.csv", "r")
line_count = 0
for line in data_file:
    data = line.strip().split(",")
    line_count = line_count + 1
    if line_count >=30000:
        epochMin = int(data[0])
        timeMap[epochMin] = {
            "epoch_day": data[1],
            "month": data[2],
            "day": data[3],
            "dow": data[4],
            "hour": data[5],
            "min": data[6],
            "headcount_unique": data[7],
            "total_wait_time": data[8],
            "device1": data[9],
            "device2": data[10],
            "device3": data[11],
            "device4": data[12],
            "device5": data[13],
            "device6": data[14],
            "device7": data[15],
            "device8": data[16],
            "predictedHeadcount": [None]*11
        }

for num in range(1, 12,1):
    file_name = "../../predicted_device_1722_1440_" + str(num) + ".csv"
    print(file_name)
    file = open(file_name, "r")
    line_count = 0
    for lines in file:
        print(file_name)
        line_count = line_count + 1
        if line_count > 1:
            extracted_data = lines.strip().split(",")
            epochMin = int(extracted_data[5])
            if epochMin in timeMap:
                timeMap[epochMin]["predictedHeadcount"][num - 1] = float(extracted_data[4])

def getNext11(epochs,index):
    return epochs[index:index+11]

def getNext(epochs,lastelem):
    mins = []
    for i in epochs:
        if (lastelem == i):
            return mins
        else:
            mins.append(i)

def getLast(epochs,lastelem):
    i = 0
    while (i < len(epochs)):
        if (lastelem == epochs[i]):
            return epochs[i+1]
        i=i+1    





epochs=[]
for epoch, data in timeMap.items():
    epochs.append(epoch)
print(len(epochs))
print(len(epochs))
index = 7

# load YAML and create model
yaml_file = open('../model_device_stateful_1_200.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = model_from_yaml(loaded_model_yaml)
# load weights into new model
loaded_model.load_weights("../model_device_stateful_1_200.h5")
print("Loaded model from disk")

predictions= DataFrame(columns=[
            'actual',
             'predicted',
              'mape',
              'mae' 
            ])
subset = []
for epoch, data in timeMap.items():
    next11 = getNext11(epochs,index)
    p=0
    begin = {}
    if (len(next11)==11):
        begin = {needed : timeMap[needed] for needed in getNext(epochs,next11[0])}
        for j in next11:
            resultset = timeMap[j]
            resultset["headcount_unique"] = resultset["predictedHeadcount"][p]
            begin[j]=resultset
            #print(j,p,resultset["headcount_unique"])
            p=p+1

        last = getLast(epochs,next11[len(next11)-1])
        #print(last)
        begin[last]=timeMap[last]
        print(len(begin))
        #for key,value in begin.items():
        #    print(key,value["headcount_unique"])
        pd = DataFrame.from_dict(begin,orient='index')
        #print(pd)
        pd.drop(columns='predictedHeadcount')

        frames=[dataset,pd]

        values = concat(frames)
        encoder = LabelEncoder()
        values=values[cols]
        #print(values.columns)
        values=values.values
        print(values.shape,dataset.shape,pd.shape)


        values[:,2] = encoder.fit_transform([int(x) for x in values[:,2]])
        values[:,3] = encoder.fit_transform([int(x) for x in values[:,3]])
        values[:,4] = encoder.fit_transform([int(x) for x in values[:,4]])
        values[:,5] = encoder.fit_transform([int(x) for x in values[:,5]])
        values[:,6] = encoder.fit_transform([int(x) for x in values[:,6]])


        values = values.astype('float32')
        # normalize features
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(values)
        scaled = scaled[30000:,]
        print(scaled.shape)
        n_hours = 3  # to t-3
        n_features = 15
        n_obs = n_hours*n_features
        reframed = convert_to_time_independent(scaled, n_hours, 1)
        values = reframed.values
        test=values
        test_X, test_y = test[:, :n_obs], test[:, -n_features]
        test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
        yhat = loaded_model.predict(test_X,batch_size=1)
        test_X = test_X.reshape((test_X.shape[0], n_hours*n_features))
        # invert scaling for forecast
        inv_yhat = concatenate((yhat, test_X[:, -14:]), axis=1)
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

        rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
        #print('Test RMSE: %.3f' % rmse)
        mape,mae = mean_absolute_percentage_error(inv_y, inv_yhat)
          
        mape[mape == 0] = np.nan
        rmses.append(rmse)
        maes.append(np.nanmean(mae))

        print('iteration :, Test RMSE:, Test MAPE:,Test MAE: %d %.3f %.3f %.3f' % (1,rmse,np.nanmean(mape),np.nanmean(mae)))

        tmmape = list(mape.flatten())


        import pandas as pd

        combined = DataFrame(
            {'actual': inv_y,
             'predicted': inv_yhat,
              'mape':pd.Series(tmmape),
              'mae' :pd.Series(mae.flatten())
            })

        predictions = predictions.append(combined.iloc[-1])
  
    index=index+1
    if (index == 2000):
        print(predictions)
        predictions.to_csv("/users/charan/predicted_device_1722_iter6.csv", sep=',')
        break










