
import sys
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

import statsmodels.api as sm
import pylab
import scipy.stats as stats
import statistics

from cassandra.cluster import Cluster
cluster = Cluster(['cass01.infra.livereachmedia.com','cass02.infra.livereachmedia.com','cass03.infra.livereachmedia.com'],port=9042)
session = cluster.connect('analytics')
site_id=2437
query= 'select device_wait_times from minutely_metrics where site_id='+str(site_id)+' and epoch_minute = 25605307 ;';


def getrecords(args):
	rows = session.execute(args)
	wait_time = dict()
	for row in rows:
		if(row.device_wait_times != []):
			wait_time=dict(row.device_wait_times)
			return wait_time.values()
			
def kmeans(cluster,wait_times):
	kmeans = KMeans(n_clusters=cluster)
	data = np.array(wait_times).reshape(-1,1)
	# Fitting the input data
	return kmeans.fit(data)

def checkBimodal(wait_times):
	cluster_map = pd.DataFrame()
	cluster_map['data_index'] = wait_times
	if len(wait_times) == 0 :
		return 
	elif len(wait_times) == 1 :
		cluster_map['cluster'] = 0
		return cluster_map
	single = kmeans(1,wait_times) 
	print("single:",single.inertia_)
	double = kmeans(2,wait_times)
	print("double: ",double.inertia_)
	mu=statistics.stdev(wait_times)
	sd=statistics.mean(wait_times)
	p_value = stats.kstest(wait_times, 'norm', args=(mu, sd)).pvalue
	print("p_value",p_value)
	# reject the null hypothesis if < 0.05 that htey come from same distribution
	clustered = single if (single.inertia_ <= double.inertia_) | (p_value > 0.05) else double
	cluster_map['cluster'] = clustered.labels_
	return cluster_map





#wait_times = getrecords(query)
#wait_times = [x for x in wait_times]

wait_times = [18, 11, 10, 6, 10, 12, 3, 3, 3, 3, 8, 17, 11, 13, 3, 7, 29, 5, 7, 9, 16, 10, 6, 9, 10, 3, 13, 12]
wait_times = [1,2,3,4,5,6,7,8,9,10]
wait_times = [10,10]
wait_times = [2,4,5,6,7,9]
wait_times = [5,5,5,5,5]

cluster_map = checkBimodal(wait_times)

print(cluster_map)



