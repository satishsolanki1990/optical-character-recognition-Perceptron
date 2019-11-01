
# coding: utf-8

# In[98]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[99]:


train = pd.read_csv('pa2_train.csv',header=None)


# In[100]:


val = pd.read_csv('pa2_valid.csv',header=None)


# In[101]:


test = pd.read_csv('pa2_test_no_label.csv',header=None)


# In[102]:


# assign labels +1 to number 3 and -1 to label 5
train[0] = np.where(train[0]==3,1,-1)
val[0] = np.where(val[0]==3,1,-1)


# In[103]:


# add bias feature
train[785] = 1
val[785] = 1
test[784] = 1


# # Part 1 : Online Perceptron

# In[104]:


y = np.transpose(np.array(train.iloc[:,0],ndmin=2))
y_val = np.transpose(np.array(val.iloc[:,0],ndmin=2))
X = np.array(train.iloc[:,1:])
X_val = np.array(val.iloc[:,1:])
d = X.shape[1]
n = X.shape[0]
n_val = X_val.shape[0]


# In[105]:


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


# In[106]:


#(a)


# In[107]:


part_1_curves = pd.DataFrame({'train':accuracy_train,'validation':accuracy_val})


# In[108]:


#part_1_curves.to_csv('part_1_curves.csv',index=False)# plot curves in report


# In[109]:


plt.plot(range(1,16),accuracy_train,range(1,16),accuracy_val)


# In[110]:


#(b)


# In[111]:


w_test = ws[13]
X_test = np.array(test)


# In[112]:


y_pred = np.sign(X_test.dot(w_test))


# In[113]:


#pd.DataFrame({'prediction':y_pred}).to_csv('oplabel.csv',index=False) # produces csv file


# # Part 2 : Average Perceptron

# In[114]:


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


# In[115]:


#(a)


# In[116]:


plt.plot(range(1,16),accuracy_train,range(1,16),accuracy_val)


# In[117]:


part_2_curves = pd.DataFrame({'train':accuracy_train,'validation':accuracy_val})
#part_2_curves.to_csv('part_2_curves.csv',index=False)# plot curves in report


# In[118]:


#(b) average is good ... best validation score = 95.3959% (best was 95.2732% with online perceptron)
#makes me think of cross validation


# In[119]:


#(c)


# In[120]:


w_test = w_bars[14]
y_pred = np.sign(X_test.dot(w_test))


# In[121]:


#pd.DataFrame({'prediction':y_pred}).to_csv('aplabel.csv',index=False) # produces csv file


# # Part 3 : Polynomial Kernel Perceptron

# In[122]:


def kernel(p):
    # gram matrices :
    temp = X.dot(np.transpose(X))+1
    K = np.power(temp,p)
    temp = X_val.dot(np.transpose(X))+1
    K_cross = np.power(temp,p)
    
    alpha = np.zeros(n)
    iters = 15
    it = 0
    accuracy_train = []
    accuracy_val = []
    alphas = []
    
    while it < iters:
        for i in range(n):
            u = (alpha*K[:,i]*y[:,0]).sum()
            alpha[u*y[:,0] <= 0] += 1
            
        accuracy_train.append((np.sign(K.dot(alpha*y[:,0]))*y[:,0]==1).sum()/n)
        accuracy_val.append((np.sign(K_cross.dot(alpha*y[:,0]))*y_val[:,0]==1).sum()/n_val)
        alphas.append(alpha)
        
        it += 1
        
    return accuracy_train,accuracy_val,alphas


# In[123]:


curves_part3 = pd.DataFrame()
for p in [1,2,3,4,5]:
    temp = kernel(p)
    curves_part3['acc_train_p_'+str(p)] = temp[0]
    curves_part3['acc_val_p_'+str(p)] = temp[1]


# ##### table :

# In[124]:


#curves_part3


# ##### what plots will look like :

# In[125]:


x = range(1,16)


# In[126]:


#plt.plot(x,curves_part3.acc_train_p_1,x,curves_part3.acc_val_p_1)


# In[127]:


#plt.plot(x,curves_part3.acc_train_p_2,x,curves_part3.acc_val_p_2)


# In[128]:


#plt.plot(x,curves_part3.acc_train_p_3,x,curves_part3.acc_val_p_3)


# In[129]:


#plt.plot(x,curves_part3.acc_train_p_4,x,curves_part3.acc_val_p_4)


# In[130]:


#plt.plot(x,curves_part3.acc_train_p_5,x,curves_part3.acc_val_p_5)


# In[131]:


# Best validation accuracies for each p


# In[132]:


table_p_part3 = pd.DataFrame({'p':[1,2,3,4,5],'Best validation accuracy':                              list(curves_part3.iloc[:,range(1,10,2)].max())})


# In[133]:


#table_p_part3


# In[134]:


#plt.plot(table_p_part3.p,table_p_part3['Best validation accuracy'])


# In[135]:


# Best alpha to generate prediction :


# In[136]:


alpha_opt = temp[2][0]
X_test = np.array(test)
temp = X_test.dot(np.transpose(X))+1
K_test = np.power(temp,5)
y_pred = np.sign(K_test.dot(alpha_opt*y[:,0]))


# In[137]:


#pd.DataFrame({'label':y_pred})#.to_csv ...

