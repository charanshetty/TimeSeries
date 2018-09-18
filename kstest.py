import statsmodels.api as sm
import pylab
import scipy.stats as stats
import statistics

x_10 = stats.norm.rvs(loc=5, scale=3, size=10)
x_50 = stats.norm.rvs(loc=5, scale=3, size=50)
x_100 = stats.norm.rvs(loc=5, scale=3, size=100)
x_1000 = stats.norm.rvs(loc=5, scale=3, size=1000)

print(x_10)
# Perform test KS test against a normal distribution with
# mean = 5 and sd = 3
wait_times = [0,1,1,2,3,3,3,4,5,5,6]
wait_times = [18, 11, 10, 6, 10, 12, 3, 3, 3, 3, 8, 17, 11, 13, 3, 7, 29, 5, 7, 9, 16, 10, 6, 9, 10, 3, 13, 12]
wait_times = [3,4,5,5,5,5,5,6,7]

mu=statistics.stdev(wait_times)
sd=statistics.mean(wait_times)
print(stats.kstest(wait_times, 'norm', args=(mu, sd)))
print(stats.kstest(x_50, 'norm', args=(5, 3)))
print(stats.kstest(x_100, 'norm', args=(5, 3)))
print(stats.kstest(x_1000, 'norm', args=(5, 3)))

from matplotlib import pyplot
pyplot.hist(x_10, color = 'blue', edgecolor = 'black',
bins = int(8))
#pyplot.plot(wait_times)
pyplot.show()