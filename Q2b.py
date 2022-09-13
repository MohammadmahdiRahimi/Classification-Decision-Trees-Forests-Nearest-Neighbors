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


def random_forest(train_data, target_column, num_of_trees, num_of_feature, max_depth): #random tree forest function
    forest = []
    feature_column = list(train_data.columns)
    feature_column.remove(target_column)
    for i in range(num_of_trees):
        bootstrap = train_data.sample(n = len(train_data), replace = True)# we make bootstrap for each iteration
        random_feature = random.choices(feature_column,k = num_of_feature)# choose random feature for each iteration
        forest.append(decision_tree(train_data, target_column, random_feature, max_depth, 0))#make list of decision trees
    
    #you can uncomment here to print each tree in order

    # for j in range(num_of_trees):
    #     print(j)
    #     printTree(forest[j])

    return forest

def predict(test_data, tree):#for this part we have new prediction function it use for just one data
    feature_column = list(test_data.columns)
    for feature in feature_column:
        if feature == tree.attribute:
            for brc in tree.branches:
                if test_data[feature].values== brc.attribute:
                    result = tree.branches[tree.branches.index(brc)]
                    if len(result.branches) == 1:
                        result = result.branches[0]

            if len(result.branches) != 0:
                return predict(test_data, result)# it is recursive
            else:
                return result.result


def forest_predict(test_data, forest, target_column):#here we repeat last prediction function for each data
    TP=0
    FP=0
    FN=0
    TN=0
    for i in range(len(test_data)):
        pos = 0
        neg = 0
        pred = -1
        test = test_data[i:i+1]
        for j in range(len(forest)):
            temp = predict(test , forest[j])
            if temp == 1:
                pos = pos +1
            elif temp == 0:
                neg = neg +1
            else:
                print('Wrong predict. Its not 1 nor 0')

        if pos > neg:# here is voting and we look at number of positive and negative and predict   
            pred = 1
        elif neg > pos:
            pred = 0
        else:
            print('voting warning')
            print('Same number of pos and neg predict by forest')
        
        if test[target_column].values == 1:
            if pred == 1:
                TP = TP + 1
            elif pred == 0:
                FN = FN +1
            else:
                print('confusion matrix warning')
                print('prediction isnt 1 or 0')
        elif test[target_column].values == 0:
            if pred == 1:
                FP = FP + 1
            elif pred == 0:
                TN = TN +1
            else:
                print('confusion matrix warning')
                print('prediction isnt 1 or 0')
        else:
            print('confusion matrix warning')
            print('test data label isnt 1 or 0')
    return TP, FP, FN, TN # it returns confusion matrix





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





target_column = 'Recidivism - Return to Prison numeric'
number_of_trees = 100 #you can change number of trees which is trained
number_of_feature = 4  #you can change number of features which is use for train
max_depth = 4   #maximum depth of each tree
forest = random_forest(train_data, target_column, number_of_trees, number_of_feature, max_depth)
cm =  forest_predict(test_data, forest, target_column)

print('Predict   1   0')
print('Actual')
print('1   ',cm[0] , '  ' ,cm[1])
print('0   ',cm[2] , '  ' ,cm[3])
#calculate accuracy
accuracy = (cm[3]+cm[0])/sum(cm)
print('Accuracy:',accuracy*100)