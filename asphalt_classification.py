import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sktime
from sktime.datasets import load_UCR_UEA_dataset
from sktime.transformations.panel.rocket import MiniRocketMultivariateVariable
import lightgbm as lgbm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

X_train, y_train = load_UCR_UEA_dataset(name='AsphaltPavementType', split='train')
X_test, y_test = load_UCR_UEA_dataset(name='AsphaltPavementType', split='test')

# plot train and test labels distribution
_, ax = plt.subplots(1, 2)
labels, counts = np.unique(y_train, return_counts=True)
ax[0].bar(labels, height=counts)
ax[0].set_title('train')

labels, counts = np.unique(y_test, return_counts=True)
ax[1].bar(labels, height=counts)
ax[1].set_title('test')
plt.savefig('class_distribution.png', dpi=300)

# Encode label
enc = LabelEncoder()
enc.fit(y_train)
y_train_tr = enc.transform(y_train)
y_test_tr = enc.transform(y_test)

# Data exploration
# print(X_train.shape) #each row is a time series
# print(type(X_train.iloc[1,0]))
# print(X_train)
# print(X_train.iloc[0,0])

ids = [0, 10, 21, 480, 500, 534, 999, 1011, 1030]
_, ax = plt.subplots(3, 3, sharex='all')
for i, a in zip(ids, ax.flat):
      arr = X_train.iloc[i, 0]
      lab = y_train[i]
      a.plot(np.arange(len(arr)), arr)
      a.set_title(lab)
plt.savefig('input_data.png', dpi=300)

# Transformation
minirocket = MiniRocketMultivariateVariable(n_jobs=-1, random_state=9)
minirocket.fit(X_train, y_train)
X_train_tr = minirocket.transform(X_train)
X_test_tr = minirocket.transform(X_test)
print(f'The transformed train dataset has shape of {X_train_tr.shape}, \n '
      f'and the transformed test dataset has shape {X_test_tr.shape}')

# Feed into a classifier
clf = lgbm.LGBMClassifier(n_estimators=500, class_weight='balanced', random_state=112, n_jobs=-1)
clf.fit(X_train_tr, y_train_tr)
y_pred = clf.predict(X_test_tr)
ConfusionMatrixDisplay.from_predictions(y_test, enc.inverse_transform(y_pred), cmap='summer', colorbar=False)
plt.title(f'Accuracy using MiniRocket: {accuracy_score(y_test_tr, y_pred)}')
plt.savefig('confusion_matrix.png', dpi=300)