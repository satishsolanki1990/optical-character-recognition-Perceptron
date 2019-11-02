
# coding: utf-8

# In[61]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[62]:


train = pd.read_csv('pa2_train.csv',header=None)


# In[63]:


val = pd.read_csv('pa2_valid.csv',header=None)


# In[64]:


test = pd.read_csv('pa2_test_no_label.csv',header=None)


# In[65]:


# assign labels +1 to number 3 and -1 to label 5
train[0] = np.where(train[0]==3,1,-1)
val[0] = np.where(val[0]==3,1,-1)


# In[66]:


# add bias feature
train[785] = 1
val[785] = 1
test[784] = 1


# # Part 1 : Online Perceptron

# In[67]:


y = np.transpose(np.array(train.iloc[:,0],ndmin=2))
y_val = np.transpose(np.array(val.iloc[:,0],ndmin=2))
X = np.array(train.iloc[:,1:])
X_val = np.array(val.iloc[:,1:])
d = X.shape[1]
n = X.shape[0]
n_val = X_val.shape[0]


# In[68]:


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


# In[69]:


#(a)


# In[70]:


part_1_curves = pd.DataFrame({'train':accuracy_train,'validation':accuracy_val})


# In[71]:


#part_1_curves.to_csv('part_1_curves.csv',index=False)# plot curves in report


# In[72]:


plt.plot(range(1,16),accuracy_train,range(1,16),accuracy_val)


# In[73]:


#(b)


# In[74]:


w_test = ws[13]
X_test = np.array(test)


# In[75]:


y_pred = np.sign(X_test.dot(w_test))


# In[76]:


#pd.DataFrame({'prediction':y_pred}).to_csv('oplabel.csv',index=False) # produces csv file


# # Part 2 : Average Perceptron

# In[77]:


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


# In[78]:


#(a)


# In[79]:


plt.plot(range(1,16),accuracy_train,range(1,16),accuracy_val)


# In[80]:


part_2_curves = pd.DataFrame({'train':accuracy_train,'validation':accuracy_val})
#part_2_curves.to_csv('part_2_curves.csv',index=False)# plot curves in report


# In[81]:


#(b) average is good ... best validation score = 95.3959% (best was 95.2732% with online perceptron)
#makes me think of cross validation


# In[82]:


#(c)


# In[83]:


w_test = w_bars[14]
y_pred = np.sign(X_test.dot(w_test))


# In[84]:


#pd.DataFrame({'prediction':y_pred}).to_csv('aplabel.csv',index=False) # produces csv file


# # Part 3 : Polynomial Kernel Perceptron

# In[85]:


def kernel(p):
    # gram matrices :
    temp = X.dot(np.transpose(X))+1
    K = np.power(temp,p)
    
    temp = X.dot(np.transpose(X_val))+1
    K_cross = np.power(temp,p)
    
    alpha = np.zeros(n)
    iters = 15
    it = 0
    accuracy_train = []
    accuracy_val = []
    alphas = []
    
    while it < iters:
        
        y_hat = np.zeros(n)
        y_hat_val = np.zeros(n_val)
        for i in range(n):
            u = (alpha*K[:,i]*y[:,0]).sum()
            if u*y[i]<=0:
                alpha[i]+=1
        
        for i in range(n):
            y_hat[i] = np.sign((K[:,i]*alpha*y[:,0]).sum())
        accuracy_train.append((y_hat*y[:,0]==1).sum()/n)
            
        for i in range(n_val):
            y_hat_val[i] = np.sign((K_cross[:,i]*alpha*y[:,0]).sum())
        accuracy_val.append((y_hat_val*y_val[:,0]==1).sum()/n_val)
            
        alphas.append(alpha)
        
        it += 1
        
    return accuracy_train,accuracy_val,alphas


# In[86]:


curves_part3 = pd.DataFrame()
for p in [1,2,3,4,5]:
    temp = kernel(p)
    curves_part3['acc_train_p_'+str(p)] = temp[0]
    curves_part3['acc_val_p_'+str(p)] = temp[1]


# ##### table :

# In[87]:


curves_part3


# In[89]:


#curves_part3.to_csv('curves_part3.csv',index=False) # plots in report


# In[92]:


curves_part3.max()


# ##### what plots will look like :

# In[93]:


x = range(1,16)


# In[95]:


plt.plot(x,curves_part3.acc_train_p_1,x,curves_part3.acc_val_p_1)


# In[96]:


plt.plot(x,curves_part3.acc_train_p_2,x,curves_part3.acc_val_p_2)


# In[97]:


plt.plot(x,curves_part3.acc_train_p_3,x,curves_part3.acc_val_p_3)


# In[98]:


plt.plot(x,curves_part3.acc_train_p_4,x,curves_part3.acc_val_p_4)


# In[99]:


plt.plot(x,curves_part3.acc_train_p_5,x,curves_part3.acc_val_p_5)


# In[100]:


# Best validation accuracies for each p


# In[101]:


table_p_part3 = pd.DataFrame({'p':[1,2,3,4,5],'Best validation accuracy':                              list(curves_part3.iloc[:,range(1,10,2)].max())})


# In[102]:


table_p_part3


# In[104]:


#table_p_part3.to_csv('table_p_part3.csv',index=False) # plot in report


# In[105]:


plt.plot(table_p_part3.p,table_p_part3['Best validation accuracy'])

