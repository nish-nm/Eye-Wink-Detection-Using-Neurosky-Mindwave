import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('humara_data_eeg_pre.csv')

del dataset['attention']
del dataset['meditation']

#dataset.drop_duplicates

#y = np.array(dataset.LOR)

columns_list = ['blinkStrength', 'delta', 'highAlpha', 'highBeta', 'highGamma', 'lowAlpha', 'lowBeta', 'lowGamma', 'theta']
X = dataset[['blinkStrength', 'delta', 'highAlpha', 'highBeta', 'highGamma', 'lowAlpha', 'lowBeta', 'lowGamma', 'theta', 'LTYRTY', 'LOR']]

X_model = np.loadtxt('X_model.txt', dtype=float)

y = np.loadtxt('y_model.txt', dtype=int)
                
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_model, y, test_size = 0.1, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

accuracy_scores = []
#BEST ONE
# Fitting Kernel SVM to the Training set
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
for n_estimators in range(100,2000,100):
    classifier = RandomForestClassifier(n_estimators = n_estimators, criterion = 'gini', random_state = 0, n_jobs=1)
    classifier.fit(X_train, y_train)
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    accuracy_scores.append([accuracy_score(y_pred, y_test)])
    
plt.plot(range(100,2000,100), accuracy_scores)    

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

accuracy_score(y_pred, y_test)

from sklearn.externals import joblib

joblib_file = "model.pkl"
joblib.dump(classifier, joblib_file)

'''
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Kernel SVM (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
'''