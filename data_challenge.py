import pickle
import pandas as pd
with open('data-challenge-student.pickle', 'rb') as handle:
    # dat = pickle.load(handle)
    dat = pd.read_pickle(handle)

X = dat['X_train']
Y = dat['Y']
S = dat['S_train']

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from evaluator import *

# Train the logistic regression
X_train, X_test, Y_train, Y_test, S_train, S_test = train_test_split(X, Y, S, test_size=0.3, random_state=42)

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

# 1. 数据预处理
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. 特征选择
selector = SelectKBest(score_func=f_classif, k=10)
X_train_selected = selector.fit_transform(X_train_scaled, Y_train)
X_test_selected = selector.transform(X_test_scaled)

# 将数据转换为 PyTorch 张量并移动到 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train_tensor = torch.tensor(X_train_selected, dtype=torch.float32).to(device)
Y_train_tensor = torch.tensor(Y_train.values, dtype=torch.long).to(device)
X_test_tensor = torch.tensor(X_test_selected, dtype=torch.float32).to(device)

# 3. 类别平衡
smote = SMOTE(random_state=42)
X_train_balanced, Y_train_balanced = smote.fit_resample(X_train_selected, Y_train)

# 将平衡后的数据转换为 PyTorch 张量并移动到 GPU
X_train_balanced_tensor = torch.tensor(X_train_balanced, dtype=torch.float32).to(device)
Y_train_balanced_tensor = torch.tensor(Y_train_5.balanced, dtype=torch.long).to(device)

# 4. 超参数调优
param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf', 'poly']}
svm_classifier = SVC(random_state=42)
grid_search = GridSearchCV(svm_classifier, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_balanced, Y_train_balanced)
best_svm = grid_search.best_estimator_

# 5. 在测试集上评估模型
y_pred = best_svm.predict(X_test_selected)
accuracy = accuracy_score(Y_test, y_pred)
print("Accuracy:", accuracy)
