# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 12:33:16 2021

@author: Nada Adel
"""

import numpy as np
import pandas as pd
import math
import copy
import csv
import time
#Global variables 
#**Filenmame**
train_filename = 'tennis.csv'
test_filename = 'test_tennis.csv'


dataset = pd.read_csv(train_filename)
X = dataset.iloc[:, 1:].values
nrows = X.shape[0]
ncols = X.shape[1]
weights={}
label_weight={}
decision_weight={}
weight_acc=1


"""
class Node represents the node of the tree
"""
class Node(object):
    def __init__(self):
        self.value = None
        self.decision = None
        self.childs = None
        self.weight=None
        
"""
 get the names of attributes of the dataset 
"""
def get_attribute(filename):
   with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter = ',')
    
        list_of_column_names = []
    
        for row in csv_reader:
            list_of_column_names.append(row)
            break
   return list_of_column_names

"""
Get distinct output labels
"""
def get_lables(filename):
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter = ',')
        labels=[]
        flag=True
        for row in csv_reader:
            if flag:
                flag=False
            else:
                labels.append(row[-1])
        
        labels=set(labels)
    return labels

"""
get the values that the attributes take
"""
def get_weights(filename):
    myWeights={}
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter = ',')  
        for row in csv_reader:
            for value in row[1::]:
                if value not in myWeights:
                    myWeights[value]=0
            
    return myWeights
"""
format test examples using dictionaries
"""
def test_dic(filename):  
    res_dic=[]
    true_labels=[]
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter = ',')   
        flag=True
        for row in csv_reader:
            if flag:
                flag=False
            else:
                record_dic={}
                inx=0
                for col in row[1:len(row)-1]:
                    if col=='':
                        record_dic[attributes[inx]]=None
                    else:
                        record_dic[attributes[inx]]=col
                    inx+=1
                res_dic.append(record_dic)
                true_labels.append(row[-1])
    return res_dic, true_labels
                
            

"""calculate base entropy"""
def findEntropy(data, rows, labels):
    N=len(rows)
    if N==0:
        return 0,-1
    frequent_labels={}
    for label in labels:
        if label not in frequent_labels:
            frequent_labels[label]=0
     
    ans = -1
    idx = len(data[0]) - 1
    entropy = 0
    # calc the frequency of each label in the dataset
    for i in rows:
        curr_label= data[i][idx]
        frequent_labels[curr_label]+=1

    entropy=0
    for key in frequent_labels:
        P_label=frequent_labels[key]/N
        if P_label==1:
            ans=key
        if frequent_labels[key]!=0:
            entropy+=P_label*math.log2(P_label)
            
    entropy*=-1
    return entropy, ans


def findMaxGain(data, rows, columns,labels):
    maxGain = 0
    retidx = -1
    entropy, ans = findEntropy(data, rows,labels)
    if entropy == 0:
        return maxGain, retidx, ans

    for j in columns:
        mydict = {}
        idx = j
        for i in rows:
            key = data[i][idx]
            if key not in mydict:
                mydict[key] = 1
            else:
                mydict[key] = mydict[key] + 1
        gain = entropy
      
        for key in mydict:
            N=0
            frequent_labels={}
            for label in labels:
              if label not in frequent_labels:
                  frequent_labels[label]=0
            

            for k in rows:
                if data[k][j] == key:
                    N+=1
                    frequent_labels[data[k][-1]]+=1
                   
            tuna=0
            for key2 in frequent_labels:
                if N!=0:
                    P_label=frequent_labels[key2]/N
                    if frequent_labels[key2]!=0:
                        tuna+=P_label*math.log2(P_label)
                        
            gain += (mydict[key] * tuna)/nrows 
    
        if gain > maxGain:
            maxGain = gain
            retidx = j

    return maxGain, retidx, ans


def buildTree(data, rows, columns,labels):

    maxGain, idx, ans = findMaxGain(X, rows, columns,labels)
    root = Node()
    root.childs = []
    if maxGain == 0:
        if ans != -1:
            root.value = ans
            return root

    root.value = attributes[idx]
    mydict = {}
    missing=False
    for i in rows:
        key = data[i][idx]
        #handle missing
        if key!=key:
            missing=True
        elif key not in mydict:
            mydict[key] = 1
        else:
            mydict[key] += 1

    newcolumns = copy.deepcopy(columns)
    if idx!=-1:
        newcolumns.remove(idx)

    for key in mydict:
        newrows = []
        for i in rows:
            if data[i][idx] == key:
                newrows.append(i)
        # print(newrows)
        temp = buildTree(data, newrows, newcolumns,labels)
        temp.decision = key
        temp.weight=weights[temp.decision]/nrows
        root.childs.append(temp)
    return root

"""
Functionality: traverse through multibranch tree 
Input:         the root(head) of the tree
Output:        traverse the tree
"""
def traverse(root):
    n = len(root.childs)

    if n > 0:
        for i in range(0, n):
            traverse(root.childs[i])

"""
Functionality: Build the tree and return the root of the tree
"""
def calculate(ncols,labels):
    rows = [i for i in range(0, nrows)]
    columns = [i for i in range(0, ncols)]
    root = buildTree(X, rows, columns,labels)
    root.decision = 'Start'
    traverse(root)
    return root

"""normal classification algorithm without handling missing values"""
def test_label(root,test):
    n = len(root.childs)
    if n > 0:
        for i in range(0, n):
            if root.value in test:  
               if test[root.value]==root.childs[i].decision:
                   test_label(root.childs[i],test)
    else:
        #print("Output Label of the test without missing value",root.value)
        label_weight[root.decision]=root.value
        decision_weight[root.decision]=root.weight

 
""" modified classification algorithm to handle missing values in test set"""         
def handle_missing(root, test,weight):
    n = len(root.childs)
    if n > 0:
        for i in range(0, n):
            if root.value in test:  
                #check if missing value
               if type(test[root.value])==type(None):
                   #print(root.value, " missing value...")
                   root.childs[i].weight*=weight
                   handle_missing(root.childs[i],test,weights[root.childs[i].decision])
               elif test[root.value]==root.childs[i].decision:
                    handle_missing(root.childs[i],test,weights[root.childs[i].decision])
    else:
        #print("Output Label of the test ",root.value," decision",root.decision, "weight ",root.weight)
        label_weight[root.decision]=root.value
        decision_weight[root.decision]=root.weight

if __name__ == '__main__':
    """mesure time"""
    startTime = time.time()

    #####your python script#####
    
    #the name of the csv file containing the dataset

    filename=train_filename
    labels=get_lables(filename)
    print(labels)
    weights=get_weights(filename)

    
    #list contains the columns in the dataset
    attributes=get_attribute(filename)[0]
    #the number of attributes in the dataset
    nfeatures=len(attributes)
    #the number of features/attributes in the dataset
    attributes=attributes[1:nfeatures-1]
    ncols=len(attributes)

    print("start building the tree ..")
    root = calculate(ncols,labels)
    #this is the test example 
    print("start testing...")
    test_file=test_filename
    #store each example record as dictionary record, the key is represented by
    #the attribute name, and the value is the value of this attribute for that record
    test_examples,true_labels=test_dic(test_file)
    correct_classified=0
    inx=0
    print("normal classification for decision tree..")
    #this function test onlyy for test example without missing values 
    for test in test_examples:
        test_label(root,test)
        
    print("modified classification for missing data...\n\n")
    for test in test_examples:   
        weight=1
        label_weight.clear()
        decision_weight.clear()
        #this function is modification to the normal classification to classify test with missing values
        handle_missing(root, test,weight)
        #output label corresponding to the highest weight 
        #test_label(root,test)
        max_weight=-1
        max_weight_label=None
        max_weight_decision=None
        for key in decision_weight:
            if decision_weight[key]>max_weight:
                max_weight=decision_weight[key]
                max_weight_label= label_weight[key]
        
        if max_weight_label==true_labels[inx]:
            correct_classified+=1
        inx+=1
        #print("\n Finally the output label is ",max_weight_label)

    executionTime = (time.time() - startTime)
    print('Execution time in seconds: ' + str(executionTime))
    print("volia! accuracy of your modified classification algorithm is ",(correct_classified/len(test_examples))*100,"%")
 



