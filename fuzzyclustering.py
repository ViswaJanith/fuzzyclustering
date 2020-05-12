# -*- coding: utf-8 -*-
"""
Created on Tue May 9 17:41:56 2020

@author: viswa janith
"""

import pandas as pd
import numpy as np
# Reading the excel file using pandas
z= pd.read_excel('Data Sets.xlsx', 'Data Set 2', header = None)
#N is length of the data
N = len(z)
#N1 is just to seperate the data
N1 = N - 20
#creating two lists 
# one for training 
# other one for testing
train_data = []
test_data = []
r = 0
# splitting Dataset into train and test
for i in range(N1):
    r = r+1
    if r % 5 == 0:
        test_data.append(z.iloc[i])
    else:
        train_data.append(z.iloc[i])
for i in range(600,620):
    test_data.append(z.iloc[i])
test_len = int(N1/5) + 20
test_index = [i for i in range(test_len)]
train_index = [i for i in range(N - test_len)]
#test dataset and train dataset
test = pd.DataFrame(test_data, index = test_index)
train = pd.DataFrame(train_data, index = train_index)
import random
import matplotlib.pyplot as plt
# Saving the test dataset to the below file
test.to_excel("testdataset_170565.xlsx")
m = 2
error = 0.000001
prev = 5
current = 0
iter_list= []
j_list = []
for c in range(2,11):
    k =[]
    print("present c-value:",c)
    for j in range (N - test_len):
        column = []
        rand = [random.randint(1,10) for k in range(c)]
        for i in range (c):
            randsum = sum(rand)
            column.append(rand[i]/randsum)
        k.append(column)
    U = np.transpose(k)

	
    
    max_u_curr= 0
    for i in U:
        for j in i:
          if(j>max_u_curr):
              max_u_curr = j
    
    iter_error = 1
    iteration = 0
    while(iter_error > error):
        v = []
        for i in range(c):
            vi = []
            numerator_x = 0
            denominator = 0
            numerator_y = 0
            for k in range(len(train_index)):
                numerator_x = numerator_x + ((U[i][k])**m) * train_data[k][0] 
                denominator = denominator + ((U[i][k])**m) 
                numerator_y = numerator_y + ((U[i][k])**m) * train_data[k][1] 
            vi.append(numerator_x/denominator)
            vi.append(numerator_y/denominator)
            v.append(vi)
        
        distances = []
        for i in range(c):
            dist = []
            for k in range(len(train_index)):
                dist.append( (train_data[k][0] - v[i][0])**2 + (train_data[k][1] - v[i][1])**2 )
            distances.append(dist)   
        for k in range(len(train_index)):
            dist_k = [i[k] for i in distances]
            if all(i>0 for i in dist_k):
                for i in range(c):
                    temp = 0
                    for j in range(c):
                        temp = temp + (dist_k[i]/dist_k[j])**(2/(m-1))
                    U[i][k] = 1/temp
            else:
                for i in range(c):
                    if dist_k[i] == 0:
                        U[i][k] = 1
                    else:
                        U[i][k] = 0
        max_u_prev = max_u_curr
        max_u_curr = 0
        for i in U:
            for j in i:
                if(j >max_u_curr):
                    max_u_curr = j
        iter_error = abs(max_u_curr - max_u_prev)
        iteration = iteration + 1
    
    iter_list.append(iteration)
#jv is value of objective function value
    jv = 0
    for i in range(c):
        for k in range(len(train_index)):
            jv = jv + ((U[i][k])**2) *((train_data[k][0] - v[i][0])**2 + (train_data[k][1] - v[i][1])**2)
    
    j_list.append(jv)
#minimur is just a variable to get min_r
minimumr = 100000
c_list = [i for i in range(2,11)]
for i in range(1,7):
    r = abs ((j_list[i]-j_list[i+1])/(j_list[i-1]-j_list[i]))  
    if r < minimumr:
        c = 4
        minimumr = r
print('No of clusters:',c)
k =[]
for j in range (N - test_len):
    column = []
    rand = [random.randint(1,10) for k in range(c)]
    for i in range (c):
        randsum = sum(rand)
        column.append(rand[i]/randsum)
    k.append(column)
U = np.transpose(k)

max_u_curr= 0
for i in U:
    for j in i:
      if(j>max_u_curr):
          max_u_curr = j

iter_error = 1
iteration = 0
while(iter_error > error):
    v = []
    for i in range(c):
        v1 = []
        numerator_x = 0
        denominator = 0
        numerator_y = 0
        for k in range(len(train_index)):
            numerator_x = numerator_x + ((U[i][k])**m) * train_data[k][0] 
            denominator = denominator + ((U[i][k])**m) 
            numerator_y = numerator_y + ((U[i][k])**m) * train_data[k][1] 
        v1.append(numerator_x/denominator)
        v1.append(numerator_y/denominator)
        v.append(v1)
    
    distances = []
    for i in range(c):
        dist = []
        for k in range(len(train_index)):
            dist.append( (train_data[k][0] - v[i][0])**2 + (train_data[k][1] - v[i][1])**2 )
        distances.append(dist)   
    for k in range(len(train_index)):
        dist_k = [i[k] for i in distances]
        if all(i>0 for i in dist_k):
            for i in range(c):
                temp = 0
                for j in range(c):
                    temp = temp + (dist_k[i]/dist_k[j])**(2/(m-1))
                U[i][k] = 1/temp
        else:
            for i in range(c):
                if dist_k[i] == 0:
                    U[i][k] = 1
                else:
                    U[i][k] = 0
    max_u_prev = max_u_curr
    max_u_curr = 0
    for i in U:
        for j in i:
            if(j >max_u_curr):
                max_u_curr = j
    iter_error = abs(max_u_curr - max_u_prev)

cluster = []
_clusters = [i for i in range(1,c+1)]
for j in range(len(train_index)):
    max_val = 0
    for i in range(c):
        if U[i][j]>max_val:
            max_val = U[i][j]
            cluster_val = i+1
    cluster.append(cluster_val)
train['class'] = cluster
col = ['x','y','class']
train.columns = col
with open('centroid.txt', 'w') as filehandle:
    for listitem in v:
        for k in listitem:
            filehandle.write('%f ' % k)
        filehandle.write('\n')
train.plot.scatter('x','y',c ='class', colormap ='jet')
       
# create figure and axis objects with subplots()
fig,ax = plt.subplots()
# make a plot
ax.plot(c_list, j_list, color="red", marker="o")
# set x-axis label
ax.set_xlabel("Cluster number",fontsize=14)
# set y-axis label
ax.set_ylabel("J-value",fontsize=14)
# twin object for two different y-axis on the sample plot
ax2=ax.twinx()
# make a plot with different y-axis using second axis object
ax2.plot(c_list, iter_list,color="blue",marker="o")
ax2.set_ylabel("Number of Iterations",color="blue",fontsize=14)
plt.show()         
           
            
            
            
