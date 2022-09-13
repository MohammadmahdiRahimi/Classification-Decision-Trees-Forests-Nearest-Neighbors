import numpy as np
import metric_learn
from metric_learn import LMNN
from metric_learn import NCA
import pandas as pd
from numpy.linalg import norm
import random

def find_distance(train_data, test_data):#this function find distance between to data 
    distance = []
    test_arr =  test_data.to_numpy()
    for i in range(len(train_data)):
        train_arr = train_data[i:i+1].to_numpy()
        temp = train_arr[0][1:14] - test_arr[0][1:14] #use all features except label
        distance.append(norm(temp,2))
    return distance

def nmax (arr, n):#this function find n nearest nighbors
    temp = np.array(arr)
    idx = (temp).argsort()[:n]
    return idx

def voting(idx, train_data):#this function is for voting and look at majority
    c1 = 0
    c2 = 0
    c3 = 0
    pred_class = 0
    for  i in range(len(idx)):#find label of nearest nighbors
        temp = train_data[idx[i]:idx[i]+1].to_numpy()
        temp_class = temp[0][0]
        if temp_class == 1:
            c1 = c1+1
        elif temp_class == 2:
            c2 = c2 +1
        elif temp_class == 3:
            c3 = c3 +1
        else:
            print('Wrong data in voting function')
    res = [c1, c2, c3]
    #look at most frequent label
    if c1 == c2 and c3 < c1:
        pred_class = random.randint(1,2)
    elif c2 == c3 and c3 > c1:
        pred_class = random.randint(2,3)
    elif c1== c3 and c3 > c2:
        ind =random.randint(0,1)
        t =np.array([3 , 2])
        pred_class =  t[ind]
    else:    
        max_value = max(res)
        max_index = res.index(max_value)
        pred_class = max_index + 1
    return pred_class
    
def knn(train_data, test_data, k):#main knn function
    a1p1 = 0    #variables for making confusion matrix
    a1p2 = 0
    a1p3 = 0
    a2p1 = 0 
    a2p2 = 0
    a2p3 = 0
    a3p1 = 0
    a3p2 = 0 
    a3p3 = 0
    for i in range(len(test_data)):#find nearest nighbor for all test data
        dis_list = find_distance(train_data, test_data[i:i+1])#find distance 
        nearest_neighbor = nmax(dis_list, k)#find nearest
        pred = voting(nearest_neighbor, train_data)#voting label
        temp = test_data[i:i+1].to_numpy()#find actual label
        actual = temp[0][0]
        if  actual == 1:#making confusion matrix
            if  pred == 1:
                a1p1 = a1p1 +1
            elif  pred == 2:
                a1p2 = a1p2 +1
            elif pred == 3:
                a1p3 = a1p3 +1
        elif actual == 2:
            if  pred == 1:
                a2p1 = a2p1 +1
            elif  pred == 2:
                a2p2 = a2p2 +1
            elif pred == 3:
                a2p3 = a2p3 +1
        elif actual == 3:
            if  pred == 1:
                a3p1 = a3p1 +1
            elif  pred == 2:
                a3p2 = a3p2 +1
            elif pred == 3:
                a3p3 = a3p3 +1
    return a1p1, a1p2, a1p3, a2p1, a2p2, a2p3, a3p1, a3p2, a3p3


#load data
_path = 'wine.csv'
df = pd.read_csv(_path)

#split data to train and test
dfr = df.copy()
dfr['randNumCol'] = np.random.uniform(0,1, len(dfr))
train_data = dfr[dfr.randNumCol>0.2]
test_data = dfr[dfr.randNumCol<=0.2]
test_data=test_data.drop(columns=['randNumCol'])
train_data=train_data.drop(columns=['randNumCol'])

#separate labels and features
x_train = train_data.drop("Class", axis=1).reset_index(drop=True)
x_test = test_data.drop("Class", axis=1).reset_index(drop=True)
y_train = train_data["Class"].reset_index(drop=True)
y_test = test_data["Class"].reset_index(drop=True)

#here we use LMNN 
lmnn = LMNN(max_iter=10000,convergence_tol=1e-4)
lmnn.fit(x_train, y_train)
lmnn_x = pd.DataFrame(lmnn.transform(x_train))
lmnn_x_test = pd.DataFrame(lmnn.transform(x_test))
lmnn_train = pd.concat([y_train, lmnn_x], axis=1)
lmnn_test = pd.concat([y_test, lmnn_x_test], axis=1)

#here we use NCA
nca = NCA(max_iter=10000,tol=1e-4)
nca.fit(x_train, y_train)
nca_x =  pd.DataFrame(nca.transform(x_train))
nca_x_test = pd.DataFrame(nca.transform(x_test))
nca_train = pd.concat([y_train, nca_x], axis=1)
nca_test = pd.concat([y_test, nca_x_test], axis=1)

k = 7
print('For ',k,' nighbor is:')

#predict with LMNN 
cm_lmnn = knn(lmnn_train,lmnn_test,k)
accuracy = (cm_lmnn[0] + cm_lmnn[4]+cm_lmnn[8])/sum(cm_lmnn)
print('Accuracy with LMNN metric = ', accuracy*100 , '%')
print('Confusion Matrix with LMNN metric:')
print('                       c1          c2          c3 ')
print('              c1      '+str(cm_lmnn[0])+ '         ' + str(cm_lmnn[1])+ '           ' +str(cm_lmnn[2]))
print('Actual:       c2       '+str(cm_lmnn[3])+ '         ' + str(cm_lmnn[4])+ '           ' +str(cm_lmnn[5]))
print('              c3       '+str(cm_lmnn[6])+ '          ' + str(cm_lmnn[7])+ '           ' +str(cm_lmnn[8]))
print('                           Predicted')

#predict with NCA
cm_nca = knn(nca_train,nca_test,k)
accuracy = (cm_nca[0] + cm_nca[4]+cm_nca[8])/sum(cm_nca)
print('Accuracy with NCA metric = ', accuracy*100 , '%')
print('Confusion Matrix with NCA metric:')
print('                       c1          c2          c3 ')
print('              c1      '+str(cm_nca[0])+ '         ' + str(cm_nca[1])+ '           ' +str(cm_nca[2]))
print('Actual:       c2       '+str(cm_nca[3])+ '         ' + str(cm_nca[4])+ '           ' +str(cm_nca[5]))
print('              c3       '+str(cm_nca[6])+ '          ' + str(cm_nca[7])+ '           ' +str(cm_nca[8]))
print('                           Predicted')
