
# coding: utf-8

# In[183]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[211]:


train = pd.read_csv('pa2_train.csv',header=None)


# In[212]:


val = pd.read_csv('pa2_valid.csv',header=None)


# In[213]:


test = pd.read_csv('pa2_test_no_label.csv',header=None)


# In[214]:


# assign labels +1 to number 3 and -1 to label 5
train[0] = np.where(train[0]==3,1,-1)
val[0] = np.where(val[0]==3,1,-1)


# In[215]:


# add bias feature
train[785] = 1
val[785] = 1
test[784] = 1


# # Part 1 : Online Perceptron

# In[216]:


y = np.transpose(np.array(train.iloc[:,0],ndmin=2))
y_val = np.transpose(np.array(val.iloc[:,0],ndmin=2))
X = np.array(train.iloc[:,1:])
X_val = np.array(val.iloc[:,1:])
d = X.shape[1]
n = X.shape[0]
n_val = X_val.shape[0]


# In[217]:


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


# In[218]:


#(a)


# In[219]:


part_1_curves = pd.DataFrame({'train':accuracy_train,'validation':accuracy_val})


# In[220]:


#part_1_curves.to_csv('part_1_curves.csv',index=False)# plot curves in report


# In[246]:


plt.plot(range(1,16),accuracy_train,range(1,16),accuracy_val)


# In[223]:


#(b)


# In[235]:


w_test = ws[13]
X_test = np.array(test)


# In[236]:


y_pred = np.sign(X_test.dot(w_test))


# In[238]:


#pd.DataFrame({'prediction':y_pred}).to_csv('oplabel.csv',index=False) # produces csv file


# # Part 2 : Average Perceptron

# In[247]:


w = np.zeros(d)
w_bar = np.zeros(d)
s = 1
iters = 15
it = 0
accuracy_train = []
accuracy_val = []
w_bars = []
while it < iters:
    for i in range(n):
        if y[i]*(np.transpose(w).dot(X[i])) <= 0:
            w += y[i]*X[i]
        w_bar = (s*w_bar+w)/(s+1)
        s += 1
    it += 1
    # Accuracies :
    accuracy_train.append((np.sign(X.dot(w_bar))*y[:,0]==1).sum()/n)
    accuracy_val.append((np.sign(X_val.dot(w_bar))*y_val[:,0]==1).sum()/n_val)
    w_bars.append(w_bar)


# In[249]:


#(a)


# In[248]:


plt.plot(range(1,16),accuracy_train,range(1,16),accuracy_val)


# In[252]:


part_2_curves = pd.DataFrame({'train':accuracy_train,'validation':accuracy_val})
#part_2_curves.to_csv('part_2_curves.csv',index=False)# plot curves in report


# In[259]:


#(b) average is good ... best validation score = 95.3959% (best was 95.2732% with online perceptron)
#makes me think of cross validation


# In[260]:


#(c)


# In[258]:


w_test = w_bars[14]
y_pred = np.sign(X_test.dot(w_test))


# In[261]:


#pd.DataFrame({'prediction':y_pred}).to_csv('aplabel.csv',index=False) # produces csv file

