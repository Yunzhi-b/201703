import numpy as np
import pandas as pd
import kernel_metrics as km
from functions_useful import KernelMultiClassifier_OneVsOne,SVMClassifier

path1 = "Xtr.csv"
path2 = "Ytr.csv"
path3 = "Xte.csv"
#  training images
Xtr = pd.read_csv(path1,header=None).dropna(axis='columns', how='all')
Ytr = pd.read_csv(path2)
#  test images
Xte = pd.read_csv(path3,header=None).dropna(axis='columns', how='all')

# Get train and test datas
X_train = Xtr.values
X_test = Xte.values
y_train = Ytr['Prediction'].values


# train the model SVM
model = KernelMultiClassifier_OneVsOne(baseClassifier=SVMClassifier,C=2.,kernel=[km.laplacian_kernel,km.hist_kernel],kernel_coef=[1.,0.0000004])
model.fit(X_train,y_train)
y_sub = model.predict(X_test)

# submission
a = pd.DataFrame()
a['Id'] = np.arange(len(y_sub))+1
a['Prediction'] = y_sub
a.to_csv('Yte.csv',index=False)
