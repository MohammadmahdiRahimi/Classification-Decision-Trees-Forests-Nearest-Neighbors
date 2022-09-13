import pandas as pd
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn import metrics

_path = 'prison_dataset.csv'
df = pd.read_csv(_path)

df = df.sample(n = len(df),random_state=0)#we shuffle data
encoder = preprocessing.LabelEncoder() #encoder for using 
encode_df = df.copy()

for attr in df.columns:  # encode string labels to numbers
    encoder.fit(df[attr])
    encode_df[attr] = encoder.transform(df[attr])
attributes = encode_df.drop("Recidivism - Return to Prison numeric", axis=1) #separate label of data and features
tags =  encode_df["Recidivism - Return to Prison numeric"]
#split train and test data
train_attributes, test_attributes, train_tags, test_tags = train_test_split(attributes, tags, test_size=0.2, random_state=0)
# train th model
rndforest =  RandomForestClassifier(criterion='entropy', max_depth=3, n_estimators=100, random_state=0)
rndforest.fit(train_attributes, train_tags)
# predict the test with our model
pred_tags = rndforest.predict(test_attributes)
#print confusion matrix and accuracy
cm = pd.crosstab(pred_tags, test_tags, rownames=['Predicted'], colnames=['Actual'])
print(cm)
print("Accuracy : ", metrics.accuracy_score(test_tags, pred_tags) * 100)