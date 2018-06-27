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

i = 1
length_X = len(X)
while i < length_X:
        temp = 1
        for j in X.iloc[i-1,:].values == X.iloc[i,:].values:
            if j == False:
                temp = 0
        if temp:
            X.drop(X.index[i], inplace=True)
        length_X = len(X)
        i += 1   

i = 0
columns_list = columns_list[:]
for i in range(0,len(X)):
    for j in range(0,len(columns_list)):
        #print(1)
        X[columns_list[j]][i] = np.fromstring(X.iloc[i,j][8:-3], sep=',')


#X = X.drop_duplicates()

#X.drop(X.index[0], inplace=True)

        
#y = X.iloc['']
y = []
X_model = []
for i in range(0,len(X)):
    temp_list = []
    if X.LOR.iloc[i] == 0:
        continue
    for columns in X.columns:
        if columns == 'LTYRTY' or columns == 'LOR':
            continue
        for j in range(0,3):    
            temp_list.append(X[columns][i][j])
    X_model.append(np.array(temp_list))
    y.append(X.LOR.iloc[i])

y = np.array(y)                 
'''                 
for i in range(0,len(X_model)):
    X_model[i].flatten()
'''                
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_model, y, test_size = 0.1, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0, verbose=True)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
accuracy_score(y_pred, y_test)

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