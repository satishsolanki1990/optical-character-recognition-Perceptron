# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 12:46:00 2019

@author: solankis
"""

import pandas as pd
import matplotlib.pyplot as plt

part1=pd.read_csv('part_1_curves.csv')
part2=pd.read_csv('part_2_curves.csv')
part3=pd.read_csv('curves_part3.csv')



fig=plt.figure()
x=range(1,16)
fig.set_size_inches(16,9)
ax=fig.add_subplot(1,1,1)
ax.xaxis.set_tick_params(labelsize=20)
ax.yaxis.set_tick_params(labelsize=20)
ax.plot(x,part1['train']*100, linewidth=3, label='Training Accuracy')
ax.plot(x,part1['validation']*100,linewidth=3, label='validation Accuracy')
plt.title('Part 1 : Accuracy vs no. of itereation',fontsize=20, fontweight='bold')
plt.xlabel('No. of iteration', fontsize=20, fontweight='bold')
plt.ylabel('Accuracy (%)',fontsize=20, fontweight='bold')
ax.legend()
plt.savefig('Part_1.png',dpi=1200)
plt.show()


## figure 
#def curves(X,name):
#    fig=plt.figure()
#    fig.set_size_inches(16,9)
#    ax=fig.add_subplot(1,1,1)
#    ax.xaxis.set_tick_params(labelsize=20)
#    ax.yaxis.set_tick_params(labelsize=20)
#    ax.plot(X*100,linewidth=3, )
#    plt.title(name+'Accuracy vs no. of itereation',fontsize=20, fontweight='bold')
#    plt.xlabel('No. of iteration', fontsize=20, fontweight='bold')
#    plt.ylabel('Accuracy (%)',fontsize=20, fontweight='bold')
#    plt.savefig(name+'.png',dpi=1200)
#    plt.show()
#    return None 
#
#
## for part 1
##curves(part1[['train'],part1['validation']],'Part 1 ')
##
##
### for part 2
##curves(part2['train'],'Part 2 Training ')
##curves(part2['validation'],'Part 2 validation ')
#
## for part 3
#for i in range(5):
#    for v in ['train','val']:
#        curves(part3['acc_'+v+'_p_'+str(i+1)],'Part 3 Kernel of order '+str(i+1)+' '+v)
#        
#
## for p vs best val accuracy
best_acc=[]
for i in range(5):
    best_acc.append(max(part3['acc_val_p_'+str(i+1)])*100)
    
fig=plt.figure()
fig.set_size_inches(16,9)
ax=fig.add_subplot(1,1,1)
ax.xaxis.set_tick_params(labelsize=20)
ax.yaxis.set_tick_params(labelsize=20)
ax.plot([1,2,3,4,5],best_acc,linewidth=3)
plt.title('Accuracy vs Kernel Order',fontsize=20, fontweight='bold')
plt.xlabel('Kernel Order (p values)', fontsize=20, fontweight='bold')
plt.ylabel('Accuracy (%)',fontsize=20, fontweight='bold')
plt.savefig('best accuracy_kernel'+'.png',dpi=1200)
plt.show()    
