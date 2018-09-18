#using fancy impute to fill null values . hadnt given gud results not sure why
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


from fancyimpute import BiScaler, KNN, NuclearNormMinimization, SoftImpute


import numpy
from sklearn.preprocessing import Imputer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


dataset = read_csv('/Users/charan/grpdminutelyMacdata_1722.csv', header=0, index_col='epoch_min')

cols=['headcount_unique','total_wait_time', 'month', 'day', 'dow', 'hour', 'min','device1','device2','device3','device4','device5','device6','device7','device8']
dataset=dataset[cols]

dataset[['device1','device2','device3'	,'device4','device5','device6','device7','device8']] = dataset[['device1','device2','device3','device4','device5','device6','device7','device8']].replace(0, numpy.NaN)

#X_filled_knn = KNN(k=5).complete(dataset)
X_filled_nnm = NuclearNormMinimization().complete(dataset)
#print(X_filled_knn)
dd = DataFrame(X_filled_nnm)
dd['epoch_min']=dataset.index
dd.columns = ['headcount_unique','total_wait_time', 'month', 'day', 'dow', 'hour', 'min','device1','device2','device3','device4','device5','device6','device7','device8','epoch_min']
dd.to_csv("/users/charan/NuclearNormMinimization.csv", sep=',')

