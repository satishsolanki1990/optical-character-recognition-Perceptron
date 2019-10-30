
# coding: utf-8

# In[183]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[184]:


train = pd.read_csv('pa2_train.csv',header=None)


# In[185]:


val = pd.read_csv('pa2_valid.csv',header=None)


# In[186]:


test = pd.read_csv('pa2_test_no_label.csv',header=None)


# In[187]:


# assign labels +1 to number 3 and -1 to label 5
train[0] = np.where(train[0]==3,1,-1)
val[0] = np.where(val[0]==3,1,-1)


# In[188]:


# add bias feature
train[785] = 1
val[785] = 1
test[784] = 1


# # Part 1 : Online Perceptron

# In[201]:


y = np.transpose(np.array(train.iloc[:,0],ndmin=2))
y_val = np.transpose(np.array(val.iloc[:,0],ndmin=2))
X = np.array(train.iloc[:,1:])
X_val = np.array(val.iloc[:,1:])
d = X.shape[1]
n = X.shape[0]
n_val = X_val.shape[0]


# In[202]:


w = np.zeros(d)
iters = 15
it = 0
accuracy_train = []
accuracy_val = []
ws = []
while it < iters:
    for i in range(n):
        if y[i]*(np.transpose(w).dot(X[i])) <= 0:
            w += y[i]*X[i]
    it += 1
    # Accuracies :
    accuracy_train.append((np.sign(X.dot(w))*y[:,0]==1).sum()/n)
    accuracy_val.append((np.sign(X_val.dot(w))*y_val[:,0]==1).sum()/n_val)
    ws.append(w)


# In[ ]:


#(a)


# In[205]:


part_1_curves = pd.DataFrame({'train':accuracy_train,'validation':accuracy_val})


# In[207]:


#part_1_curves.to_csv('part_1_curves.csv',index=False)# plot curves in report


# In[209]:


dff=pd.read_csv('part_1_curves.csv')


# In[210]:


dff


# In[ ]:


results_part1 = pd.DataFrame({'Gamma':gammas,'iterations':all_c,'SSE training':final_SSEs_train,             'SSE validation':SSEs_validation,'MRAE':all_MRAE})
#results_part1.to_excel('results_part1.xlsx',index=False)# in report


# In[ ]:


#test[['id','predicted_price']].to_csv('prediction.csv',index=False)# .csv file sent


# In[203]:


plt.plot(accuracy_train)


# In[204]:


plt.plot(accuracy_val)


# In[ ]:


def learn(train,dev,gamma,max_it,lambdA):
    t = time.time()
    y = np.transpose(np.array(train.normalized_price,ndmin=2))
    y_raw = np.transpose(np.array(train.price,ndmin=2))
    X = np.array(train.drop(['price','normalized_price'],axis=1))
    features = list(train.drop(['price','normalized_price'],axis=1).columns)
    N = X.shape[0]
    d = X.shape[1]
    w = np.random.rand(d,1)
    
    norm_grad = 100
    SSE = []
    c = 0
    while (norm_grad > eps) & (c < max_it):
        error = X.dot(w) - y
        grad = 2*np.transpose(X).dot(error)+2*lambdA*w
        norm_grad = np.linalg.norm(grad)
        w -= gamma*grad
        SSE.append(np.linalg.norm(((X.dot(w))*(M-m)+m) - y_raw)**2)
        c += 1
    # SSE validation :
    y_dev = np.transpose(np.array(dev.price,ndmin=2))
    X_dev = np.array(dev.drop('price',axis=1))
    SSE_dev = X_dev.dot(w)
    SSE_dev = np.linalg.norm(((X_dev.dot(w))*(M-m)+m) - y_dev)**2
    
    # Mean Relative Absolute Error on validation :
    MRAE = np.round((pd.Series((((X_dev.dot(w))*(M-m)+m) - y_dev)[:,0]).map(abs)/dev.price).mean(),4)
    
    elapsed = time.time() - t
    return (w,SSE,c,elapsed,SSE_dev,MRAE,features)

