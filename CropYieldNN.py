# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv('Crop_recommendationNoPHK.csv')
df.head()

df.describe()

# sns.heatmap(df.isnull(),cmap="coolwarm")
# plt.show()

# plt.figure(figsize=(12,5))
# plt.subplot(1, 2, 1)
# # sns.distplot(df_setosa['sepal_length'],kde=True,color='green',bins=20,hist_kws={'alpha':0.3})
# sns.distplot(df['temperature'],color="purple",bins=15,hist_kws={'alpha':0.2})
# plt.subplot(1, 2, 2)
# sns.distplot(df['ph'],color="green",bins=15,hist_kws={'alpha':0.2})

# sns.countplot(y='label',data=df, palette="plasma_r")

# sns.pairplot(df, hue = 'label')

# sns.pairplot(df, hue = 'label')

# sns.jointplot(x="K",y="N",data=df[(df['N']>40)&(df['K']>40)],hue="label")

# sns.jointplot(x="K",y="humidity",data=df,hue='label',size=8,s=30,alpha=0.7)

# sns.boxplot(y='label',x='ph',data=df)

# sns.boxplot(y='label',x='P',data=df[df['rainfall']>150])

# sns.lineplot(data = df[(df['humidity']<65)], x = "K", y = "rainfall",hue="label")

c=df.label.astype('category')
targets = dict(enumerate(c.cat.categories))
df['target']=c.cat.codes

y=df.target
X=df[['temperature','humidity']]

# sns.heatmap(X.corr())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=1)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)

# we must apply the scaling to the test set as well that we are computing for the training set
X_test_scaled = scaler.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train_scaled, y_train)
knn.score(X_test_scaled, y_test)

from sklearn.metrics import confusion_matrix
mat=confusion_matrix(y_test,knn.predict(X_test_scaled))
df_cm = pd.DataFrame(mat, list(targets.values()), list(targets.values()))
# sns.set(font_scale=1.0) # for label size
# plt.figure(figsize = (12,8))
# sns.heatmap(df_cm, annot=True, annot_kws={"size": 12},cmap="terrain")

k_range = range(1,11)
scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train_scaled, y_train)
    scores.append(knn.score(X_test_scaled, y_test))



# plt.xlabel('k')
# plt.ylabel('accuracy')
# plt.scatter(k_range, scores)
# plt.vlines(k_range,0, scores, linestyle="dashed")
# plt.ylim(0.96,0.99)
# plt.xticks([i for i in range(1,11)]);

from sklearn.svm import SVC

svc_linear = SVC(kernel = 'linear').fit(X_train_scaled, y_train)
print("Linear Kernel Accuracy: ",svc_linear.score(X_test_scaled,y_test))

svc_poly = SVC(kernel = 'rbf').fit(X_train_scaled, y_train)
print("Rbf Kernel Accuracy: ", svc_poly.score(X_test_scaled,y_test))

svc_poly = SVC(kernel = 'poly').fit(X_train_scaled, y_train)
print("Poly Kernel Accuracy: ", svc_poly.score(X_test_scaled,y_test))

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

parameters = {'C': np.logspace(-3, 2, 6).tolist(), 'gamma': np.logspace(-3, 2, 6).tolist()}
# 'degree': np.arange(0,5,1).tolist(), 'kernel':['linear','rbf','poly']

model = GridSearchCV(estimator = SVC(kernel="linear"), param_grid=parameters, n_jobs=-1, cv=4)
model.fit(X_train, y_train)

print(model.best_score_ )
print(model.best_params_ )

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
clf.score(X_test,y_test)

# plt.figure(figsize=(10,4), dpi=80)
# c_features = len(X_train.columns)
# plt.barh(range(c_features), clf.feature_importances_)
# plt.xlabel("Feature importance")
# plt.ylabel("Feature name")
# plt.yticks(np.arange(c_features), X_train.columns)
# plt.show()

'''
max depth and n_estimator are important to fine tune otherwise trees will be densely graphed which will be a classic case of overfitting. max_depth=4 and n_estimators=10 gives pretty much satisfying results by making sure model is able to generalize well.
'''

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=4,n_estimators=100,random_state=42).fit(X_train, y_train)

print('RF Accuracy on training set: {:.2f}'.format(clf.score(X_train, y_train)))
print('RF Accuracy on test set: {:.2f}'.format(clf.score(X_test, y_test)))

from yellowbrick.classifier import ClassificationReport
classes=list(targets.values())
# visualizer = ClassificationReport(clf, classes=classes, support=True,cmap="Blues")

# visualizer.fit(X_train, y_train)  # Fit the visualizer and the model
# visualizer.score(X_test, y_test)  # Evaluate the model on the test data
# visualizer.show()

from sklearn.ensemble import GradientBoostingClassifier
grad = GradientBoostingClassifier().fit(X_train, y_train)
print('Gradient Boosting accuracy : {}'.format(grad.score(X_test,y_test)))
#print(n:=grad.predict([[24, 81]])) #DATA POINT
humidity=float(input("humidity: "))
temperature=float(input("temperature: "))
n=grad.predict([[temperature, humidity]])

p = ['rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas', 'mothbeans', 'mungbean', 'blackgram', 'lentil', 'pomegranate', 'banana', 'mango', 'grapes', 'watermelon', 'muskmelon', 'apple', 'orange', 'papaya', 'coconut', 'cotton', 'jute', 'coffee']
print(p[n[0]])



###########################################################################
print('')
print('Optimal neural network, uses soil nutrients')
df=pd.read_csv('Crop_recommendation.csv')
df.head()

df.describe()

# sns.heatmap(df.isnull(),cmap="coolwarm")
# plt.show()

# plt.figure(figsize=(12,5))
# plt.subplot(1, 2, 1)
# # sns.distplot(df_setosa['sepal_length'],kde=True,color='green',bins=20,hist_kws={'alpha':0.3})
# sns.distplot(df['temperature'],color="purple",bins=15,hist_kws={'alpha':0.2})
# plt.subplot(1, 2, 2)
# sns.distplot(df['ph'],color="green",bins=15,hist_kws={'alpha':0.2})

# sns.countplot(y='label',data=df, palette="plasma_r")

# sns.pairplot(df, hue = 'label')

# sns.pairplot(df, hue = 'label')

# sns.jointplot(x="K",y="N",data=df[(df['N']>40)&(df['K']>40)],hue="label")

# sns.jointplot(x="K",y="humidity",data=df,hue='label',size=8,s=30,alpha=0.7)

# sns.boxplot(y='label',x='ph',data=df)

# sns.boxplot(y='label',x='P',data=df[df['rainfall']>150])

# sns.lineplot(data = df[(df['humidity']<65)], x = "K", y = "rainfall",hue="label")

c=df.label.astype('category')
targets = dict(enumerate(c.cat.categories))
df['target']=c.cat.codes

y=df.target
X=df[['N','P','K','temperature','humidity','ph','rainfall']]

# sns.heatmap(X.corr())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=1)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)

# we must apply the scaling to the test set as well that we are computing for the training set
X_test_scaled = scaler.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train_scaled, y_train)
knn.score(X_test_scaled, y_test)

from sklearn.metrics import confusion_matrix
mat=confusion_matrix(y_test,knn.predict(X_test_scaled))
df_cm = pd.DataFrame(mat, list(targets.values()), list(targets.values()))
# sns.set(font_scale=1.0) # for label size
# plt.figure(figsize = (12,8))
# sns.heatmap(df_cm, annot=True, annot_kws={"size": 12},cmap="terrain")

k_range = range(1,11)
scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train_scaled, y_train)
    scores.append(knn.score(X_test_scaled, y_test))



# plt.xlabel('k')
# plt.ylabel('accuracy')
# plt.scatter(k_range, scores)
# plt.vlines(k_range,0, scores, linestyle="dashed")
# plt.ylim(0.96,0.99)
# plt.xticks([i for i in range(1,11)]);

from sklearn.svm import SVC

svc_linear = SVC(kernel = 'linear').fit(X_train_scaled, y_train)
print("Linear Kernel Accuracy: ",svc_linear.score(X_test_scaled,y_test))

svc_poly = SVC(kernel = 'rbf').fit(X_train_scaled, y_train)
print("Rbf Kernel Accuracy: ", svc_poly.score(X_test_scaled,y_test))

svc_poly = SVC(kernel = 'poly').fit(X_train_scaled, y_train)
print("Poly Kernel Accuracy: ", svc_poly.score(X_test_scaled,y_test))

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

parameters = {'C': np.logspace(-3, 2, 6).tolist(), 'gamma': np.logspace(-3, 2, 6).tolist()}
# 'degree': np.arange(0,5,1).tolist(), 'kernel':['linear','rbf','poly']

model = GridSearchCV(estimator = SVC(kernel="linear"), param_grid=parameters, n_jobs=-1, cv=4)
model.fit(X_train, y_train)

print(model.best_score_ )
print(model.best_params_ )

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
clf.score(X_test,y_test)

# plt.figure(figsize=(10,4), dpi=80)
# c_features = len(X_train.columns)
# plt.barh(range(c_features), clf.feature_importances_)
# plt.xlabel("Feature importance")
# plt.ylabel("Feature name")
# plt.yticks(np.arange(c_features), X_train.columns)
# plt.show()

'''
max depth and n_estimator are important to fine tune otherwise trees will be densely graphed which will be a classic case of overfitting. max_depth=4 and n_estimators=10 gives pretty much satisfying results by making sure model is able to generalize well.
'''

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=4,n_estimators=100,random_state=42).fit(X_train, y_train)

print('RF Accuracy on training set: {:.2f}'.format(clf.score(X_train, y_train)))
print('RF Accuracy on test set: {:.2f}'.format(clf.score(X_test, y_test)))

from yellowbrick.classifier import ClassificationReport
classes=list(targets.values())
# visualizer = ClassificationReport(clf, classes=classes, support=True,cmap="Blues")

# visualizer.fit(X_train, y_train)  # Fit the visualizer and the model
# visualizer.score(X_test, y_test)  # Evaluate the model on the test data
# visualizer.show()

from sklearn.ensemble import GradientBoostingClassifier
grad = GradientBoostingClassifier().fit(X_train, y_train)
print('Gradient Boosting accuracy : {}'.format(grad.score(X_test,y_test)))