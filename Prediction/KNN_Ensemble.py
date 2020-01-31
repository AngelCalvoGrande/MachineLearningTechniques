import pandas as pd

# If using centroids Dataset
spy = pd.read_csv('finalDataset.csv')

spy = spy.sample(frac=1) # Shuffle data

p_train = 0.80 
train = spy[:int((len(spy))*p_train)]
test = spy[int((len(spy))*p_train):]

print("Training samples ", len(train))
print("Test Samples: ", len(test))

features = spy.columns.difference(['attack'])
x_train = train[features]
y_train = train['attack']

x_test = test[features]
y_test = test['attack']

X, y = x_train, y_train 

import numpy as np
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

clf_rf = RandomForestClassifier(n_estimators = 512, criterion = 'entropy', 
                                max_depth=None, max_features = 'auto', 
                                min_samples_leaf = 3, min_samples_split = 4,
                                bootstrap=True, n_jobs=-1, 
                                class_weight=None)

clf_rf.fit(x_train, y_train) 

preds_rf = clf_rf.predict(x_test) # Test del modelo

from sklearn.metrics import classification_report

print("Random Forest: \n" 
      +classification_report(y_true=test['attack'], y_pred=preds_rf))

# Confussion Matrix

print("Confussion Matrix:\n")
matriz = pd.crosstab(test['attack'], preds_rf, rownames=['actual'], colnames=['preds'])
print(matriz)

# Variables relevantes

print("Feature Relevance:\n")
print(pd.DataFrame({'Feature': features ,
              'Relevancy': clf_rf.feature_importances_}),"\n")
print("Maximum relevance RF :" , max(clf_rf.feature_importances_), "\n")
