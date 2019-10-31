# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 18:16:08 2019

@author: solankis
"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# read data files

train = pd.read_csv('pa2_train.csv',header=None)

val = pd.read_csv('pa2_valid.csv',header=None)

test = pd.read_csv('pa2_test_no_label.csv',header=None)


# assign labels +1 to number 3 and -1 to label 5
train[0] = np.where(train[0]==3,1,-1)
val[0] = np.where(val[0]==3,1,-1)


# add bias feature
train[785] = 1
val[785] = 1
test[784] = 1


X = np.array(train.iloc[:,1:])
X_val = np.array(val.iloc[:,1:])
d = X.shape[1]
n = X.shape[0]
n_val = X_val.shape[0]



## gram matrix

def gram(p,X):
    
    n = X.shape[0]
    K=np.empty([n,n])
    for i in range(n):
        for j in range(i,n):
            K[i,j]=(1+X[j].dot(np.transpose(X[i])))**p
            K[j,i]= K[i,j]
    return K


# # Part 3 : Kernal Perceptron
p=[1,2,3,4,5]
y = np.transpose(np.array(train.iloc[:,0],ndmin=2))
y_val = np.transpose(np.array(val.iloc[:,0],ndmin=2))


#w = np.zeros(d)
iters = 15
it = 0
accuracy_train = []
#accuracy_val = []
#ws = []
alpha=np.zeros(n)
y_hat=[]

for a in p:    
    print("kernel",a)
    K=gram(a,X)
    while it < iters:
        for i in range(n):
            # u = prediction of y
            u=sum([alpha[j]*K[j,i]*y[j] for j in range(n)])

            if it==iters:
                y_hat.append(np.sign(u))
                
            if u*y[i] <= 0:
                alpha[i]+=1
            
        it += 1
        print('iteration:',it)
        # Accuracies :

        accuracy_train.append((y_hat[:]*y[:]==1).sum()/n)
#        accuracy_val.append((np.sign(X_val.dot(alpha))*y_val[:,0]==1).sum()/n_val)
#        ws.append(alpha)
        
    part_3_curves = pd.DataFrame({'train':accuracy_train})#'validation':accuracy_val})
    part_3_curves.to_csv('part_3_curves_'.join(str(a))+'_.csv',index=False)# plot curves in report

# dff=pd.read_csv('part_3_curves.csv')


#results_part1 = pd.DataFrame({'Gamma':gammas,'iterations':all_c,'SSE training':final_SSEs_train,             'SSE validation':SSEs_validation,'MRAE':all_MRAE})
#results_part1.to_excel('results_part1.xlsx',index=False)# in report

#test[['id','predicted_price']].to_csv('prediction.csv',index=False)# .csv file sent

    plt.figure(a)
    plt.plot(accuracy_train)
#    plt.plot(accuracy_val)
    
