import math
import numpy as np
import pandas as pd
import random
def Entropy (p,n):      #function for calculate entropy
    t = p+n
    if n == 0:
        e = -(p/t)*math.log2(p/t)       #for Exception
    elif p == 0:
        e = -(n/t)*math.log2(n/t)
    else:
        e = -(p/t)*math.log2(p/t) -(n/t)*math.log2(n/t)     #use formula to calculate
    return e

def gain (ps , ns):     #function for calculate gain
    if len(ps) != len(ns):        #it check dimension of inputs
        print("Input must be in same dimension")
        return
    t = np.sum(ps) + np.sum(ns) 
    e = Entropy(np.sum(ps),np.sum(ns))
    g = e
    for i in range(len(ps)):
        etemp = Entropy(ps[i],ns[i])
        g = g - ((ps[i]+ns[i])/t)*etemp
    return g

def findnode(df, headers, target_column):       #this function find best feature beteween list of features actually find feature which have best gain
    gs = []
    for attribute in headers:       #loop for search features
        temp = df[attribute].unique()
        p = []
        n = []
        for i in temp:
            p.append(len(df[(df[attribute]==i)&(df[target_column]==1)]))
            n.append(len(df[(df[attribute]==i)&(df[target_column]==0)]))
        tempg = gain(p,n)
        gs.append(tempg)
    nodecandidatevalue = max (gs)
    nodecandidateindex = gs.index(nodecandidatevalue)
    return headers[nodecandidateindex],df[headers[nodecandidateindex]].unique()

class layer:
    def __init__(self):
        self.result = ""
        self.attribute = ""
        self.branches = []
        self.isEnd = False
        self.depth = 0

def decision_tree(df, target_column, feature_column, max_depth, last_depth):
    first_point = layer()
    first_point.depth = last_depth
    candidate_feature, values_list = findnode(df, feature_column, target_column)
    first_point.attribute = candidate_feature
    if len(feature_column) == 0:
        return first_point
    for value in values_list:
        group = df[df[candidate_feature] == value]
        # Stop if entropy is 0
        p = len(group[group[target_column]==1])
        n = len(group[group[target_column]==0])
        entropy = Entropy(p, n)
        
        if (entropy == 0) or (first_point.depth == max_depth -1) :
            next_point = layer()
            next_point.attribute = value
            next_point.isEnd = True
            lable = group[target_column].mode()
            if len(lable)==2:
                next_point.result = random.randint(0,1)
            else:
                next_point.result = int(lable.values)
            first_point.branches.append(next_point)
            next_point.depth = first_point.depth +1
            
        # Calculate tree if entropy is not 0
        else:
            temp_point = layer()
            temp_point.attribute = value
            temp_point.depth = first_point.depth +1
            temp_feature = feature_column.copy()
            temp_feature.remove(candidate_feature)
            recursive = decision_tree(group, target_column, temp_feature, max_depth, temp_point.depth )
            temp_point.branches.append(recursive)
            first_point.branches.append(temp_point)
    return first_point

def printTree(root: layer, depth=0):#function for printing tree
    for i in range(depth):
        print("\t", end="")
    print(root.attribute, end="")
    if root.isEnd:
        print(" -> ", root.result)
    print()
    for child in root.branches:
        printTree(child, depth + 1)

def predict(decision_tree: layer, data, target_column):#function use for prediction
    headers = list(data.columns)
    TP=0
    FP=0
    FN=0
    TN=0
    if decision_tree.isEnd:# when its end of our tree or in node
        if decision_tree.result == 1:
            TP = len(data[data[target_column]==1] )
            FN = len(data[data[target_column]==0] )
        elif decision_tree.result == 0:
            FP = len(data[data[target_column]==1] )
            TN = len(data[data[target_column]==0] )
    else:
        if len(decision_tree.branches) > 1:
            for i in range(len(decision_tree.branches)):
                if  len(decision_tree.branches[i].branches) != 1:
                    new_data = data[data[decision_tree.attribute]==decision_tree.branches[i].attribute]
                    new_tree = decision_tree.branches[i]
                    [TP , FP , FN, TN] = [TP , FP , FN, TN] + np.array(predict(new_tree,new_data,target_column))
                elif len(decision_tree.branches[i].branches) == 1:
                    new_data = data[data[decision_tree.attribute]==decision_tree.branches[i].attribute]
                    new_tree = decision_tree.branches[i]
                    [TP , FP , FN, TN] = [TP , FP , FN, TN] + np.array(predict(new_tree,new_data,target_column))
        else:
            new_tree = decision_tree.branches[0]
            [TP , FP , FN, TN] = [TP , FP , FN, TN] + np.array(predict(new_tree,data,target_column))
    return TP, FP, FN, TN




#Here we load data on dataframe
_path = 'prison_dataset.csv'
df = pd.read_csv(_path)

#Split data to train and test
dfr = df.copy()
dfr['randNumCol'] = np.random.uniform(0,1, len(dfr))    #we add ne column to dataframe which is random uniform number
test_data = dfr[dfr.randNumCol<=0.2]    #it Approximately split 80% of data for train
train_data = dfr[dfr.randNumCol>0.2]
test_data = test_data.drop(columns=['randNumCol'])   #remove column which added
train_data = train_data.drop(columns=['randNumCol'])



max_depth = 5     #you can change depth
target_column = 'Recidivism - Return to Prison numeric'     #it is feature which we predict on it
feature_column = list(train_data.columns)
feature_column.remove(target_column)
tree = decision_tree(train_data, target_column, feature_column, max_depth,0)        #use function to make decision tree


printTree(tree)     #you can print decision tree or you can comment it


cm = predict(tree,test_data,target_column)       #use function to predict it gives you TP FP FN TN
#print confusion matrix
print('With depth = ', max_depth , 'we have:')
print('Predict   1   0')
print('Actual')
print('1   ',cm[0] , '  ' ,cm[1])
print('0   ',cm[2] , '  ' ,cm[3])
#calculate accuracy
accuracy = (cm[3]+cm[0])/sum(cm)
print('Accuracy:',accuracy*100, '%')