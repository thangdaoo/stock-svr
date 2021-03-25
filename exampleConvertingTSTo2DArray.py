##In this Python script, we will convert a time series of data into a 2D numpy.array (x) where every 
##  each "row" is the three-day lagged data for a specific date, and each column corresponds to
##  a lag.

import numpy as np

#This array represents all the data in our time series
timeSeries = np.array([1,2,3,4,5,6,7,8,9])

#Here we are going to initialize our data frame for computational speed and ease
lag = 3
x = np.zeros((len(timeSeries) - lag + 1,lag))

#This "counter" variable will store what row we are filling in
rowNumber = 0

#The iterator on our for loop will indicate what is the array index for the first day that should *not*
#   be included in the row. Alternatively, we could have created two iterators, the second corresponding to
#   row number and joined the necessary Iterable objects with a call to zip (if you are familar with what that
#   means).
for i in range(lag,len(timeSeries)+1):
    x[rowNumber,] = timeSeries[(i-lag):(i)]
    print(x)
    rowNumber += 1
    print()


