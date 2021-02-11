#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#including libraries
import numpy as np                          #scientific computing
import numpy.random as ran 
import matplotlib.pyplot as plt             #plot library
import csv                                  #import csv data
from IPython.core.debugger import set_trace #debugging
import pandas as pd                         #import data
from scipy.io import loadmat                #import mat data

#1 GENERATE DATA/ DRAW FROM PROBABILITY DISTRIBUTIONS
#distribution properties:
nsamples=100000         #number of samples drawn
sigma=np.sqrt(3)        #standard deviation
mu=2
mu_p= 5

#print('program stopped.')
#set_trace()
type(sigma)

#generate arrays:
x1 = np.ndarray((nsamples,1))
x2 = np.ndarray((nsamples,1))

##generate data by loop:
#for i in range(0,nsamples-1): #indexing starts with 0!!!
 #   x1[i] = ran.normal(mu, sigma)   
 #   x2[i] = ran.poisson(mu_p)
    
x1=np.random.normal(mu,sigma,nsamples)    #draw from normal dist
x2=ran.poisson(mu_p,nsamples)       #draw from poisson dist

str1='x1 has length '
str2=str(len(x1))
print(str1+str2)

str1='The total number of elements in x1 is '
str2=str(np.size(x1))
print(str1+str2)

#2 GENERATE FIGURES
#plot histogram:
fig1 = plt.subplots()
plt.subplot(1,2,1)
binwidth = 0.5   
plt.hist(x1, bins=np.arange(min(x1), max(x1) + binwidth, binwidth))
#plt.hist(x1)
plt.title("Gauss histogram" )
plt.xlabel("x")
plt.ylabel("#")

plt.subplot(1,2,2)
#binwidth2 = 1   
#plt.hist(x2, bins=np.arange(min(x2), max(x2) + binwidth2, binwidth2))
plt.hist(x2)
plt.title("Poisson histogram")
plt.xlabel("x")
plt.ylabel("#")
plt.show(fig1)


#3 LOAD EXTERNAL DATA
#load .csv file

# VERSION 01
data = []
filename='data_example.csv'
with open(filename) as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',')
    for d in csvreader:
        data.append(d)
data.pop(0)  # first row in csv file is "x","y"
data = [[float(x), float(y)] for (x,y) in data]

#data[0:5] #shows list elements 0 to 5
#tmp=data[0] #creates new list with first element of data
#tmp1=tmp[0]
#tmp2=tmp[1]

#VERSION 02 - using pandas library
filename='sunspotData.csv'
data2 = pd.read_csv(filename, sep='\t').values[:,0]

data2 = pd.read_csv(filename) #pandas object
data2 = np.array(data2)[:, 0]

data2 = pd.read_csv('sunspotData.csv').values[:,0] #numpy array

#data = pd.read_excel('investment.xls').values[:,0]


#load .mat file
mat_file = loadmat('sunspotData.mat')
sunspots=mat_file["sunspots"][:,:]
print(data1.size)
years=mat_file["years"][:,:]
