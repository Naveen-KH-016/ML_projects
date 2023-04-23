# Jesus is my Saviour!
import os
os.chdir('C:\\Users\\Dr Vinod\\Desktop\\WD_python')
import pandas as pd 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import chi2_contingency
from sklearn.preprocessing import LabelEncoder 
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.utils import resample

df = pd.read_csv('hnc.csv') #19642; 31 , 1st is unique id 
df.info()

#_______________VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor

# first put your predictors in x
x = df.iloc[:, [1,2,15,16,17,18,19,20,21,23,26]] # x is a data frame
# VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = x.columns 
  
# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(x.values, i)
                          for i in range(len(x.columns))]
  
print(vif_data) # two columns, feature & VIF will appear
'''
 feature        VIF
0               h1n1_worry   5.759713
1           h1n1_awareness   6.628330
2   is_h1n1_vacc_effective  21.390106
3            is_h1n1_risky   7.203521
4      sick_from_h1n1_vacc   4.689096
5   is_seas_vacc_effective  20.908152
6            is_seas_risky   8.017402
7              age_bracket   5.406338
8            qualification  13.411080
9             income_level  16.792285
10              employment  15.695942
'''
# drop VIF> 10
df = df.drop(['is_h1n1_vacc_effective', 'is_seas_vacc_effective',
               'qualification', 'employment'], axis = 1) 
df.info() # 19642, 27 [with 1st as Unnamed: 0, lets remove this]
df = df.drop(['Unnamed: 0'], axis = 1)
df.info() # 19642, 26
df.to_csv('hnvif.csv')
# X and y
X = df.loc[:, df.columns != 'h1n1_vaccine']
y = df.loc[:, df.columns == 'h1n1_vaccine']

# solver = liblinear
'''
liblinear [library for linear classification]: good for small data
newton-cg [newton conjugate]: can be used in this case
lbfgs[limited memory BFGS]: for multiclass problems
BFGS:Broyden–Fletcher–Goldfarb–Shanno algorithm 
sag [Stochastic Average Gradient Descent]: good for large data sets
saga: a little variant of sag
'''
from sklearn.linear_model import LogisticRegression
model1 = LogisticRegression(solver='liblinear', random_state=0)
model1.fit(X, y)
model1.intercept_
model1.coef_

#Predictions
y_pred = model1.predict(X)

#Confusion matrix
from sklearn import metrics
cm = metrics.confusion_matrix(y, y_pred)
print(cm)
'''
[[14200   928]
 [ 2608  1906]]
'''
#Accuracy Score - correct predictions / total number of data points
model1.score(X,y) #.0.82
(14200+1906)/(14200+928+2608+1906) # 0.82

#Classification report
from sklearn.metrics import classification_report
print(classification_report(y, y_pred))

#ROC Curve - Receiver Operating Characteristic curve
#tpr = True Positive Rate 
#fpr = False Positive Rate
from sklearn.metrics import roc_curve, auc, roc_auc_score
y_pred_prob = model1.predict_proba(X)
fpr, tpr, thresholds =roc_curve(df["h1n1_vaccine"], y_pred_prob[:,1])
roc_auc = auc(fpr, tpr) #Area under Curve 0.82
print(roc_auc)

#ROC Curve
plt.title('ROC Curve for LogReg: liblinear')
plt.xlabel('False Positive Rate (1-Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.plot(fpr, tpr, label = 'AUC =' +str(roc_auc))
plt.legend(loc=4) #Location of label
plt.show()

#____________________SMOTE 

novac = df[df.h1n1_vaccine == 0] #15128,26
vac = df[df.h1n1_vaccine == 1] #4514, 26
#__________________________________ oversample minority_with replacement
from sklearn.utils import resample
vac_oversample = resample(vac,
                          replace=True, # sample with replacement
                          n_samples=len(novac), # match number in majority class
                          random_state=27) # reproducible results

# combine majority and oversampled minority

dfsmote = pd.concat([novac, vac_oversample]) 
dfsmote.h1n1_vaccine.value_counts()
'''
1    15128
0    15128
'''

dfsmote.to_csv('hnsmote.csv')
#___________________lets re do log reg
# X and y
X2 = dfsmote.loc[:, dfsmote.columns != 'h1n1_vaccine']
y2 = dfsmote.loc[:, dfsmote.columns == 'h1n1_vaccine']
y2.value_counts() # both 15,128

# solver = liblinear
'''
liblinear [library for linear classification]: good for small data
newton-cg [newton conjugate]: can be used in this case
lbfgs[limited memory BFGS]: for multiclass problems
BFGS:Broyden–Fletcher–Goldfarb–Shanno algorithm 
sag [Stochastic Average Gradient Descent]: good for large data sets
saga: a little variant of sag
'''
from sklearn.linear_model import LogisticRegression
model2 = LogisticRegression(solver='liblinear', random_state=0)
model2.fit(X2, y2)
model2.intercept_
model2.coef_

#Predictions
y_pred2 = model2.predict(X2)

#Confusion matrix
from sklearn import metrics
cm2 = metrics.confusion_matrix(y2, y_pred2)
print(cm2)
'''
WITH SMOTE=
[[11848  3280]
 [ 4154 10974]]

WITHOUT SMOTE = 
[[14200   928]
 [ 2608  1906]]
'''
#Accuracy Score - correct predictions / total number of data points
model2.score(X2,y2) #WITH SMOTE = 0.75; without = #.0.82
(11848+10974)/(11848+3200+4154+10974) # 0.75

#Classification report
from sklearn.metrics import classification_report
print(classification_report(y2, y_pred2))

#ROC Curve - Receiver Operating Characteristic curve
#tpr = True Positive Rate 
#fpr = False Positive Rate
from sklearn.metrics import roc_curve, auc, roc_auc_score
y_pred_prob2 = model2.predict_proba(X2)
fpr2, tpr2, thresholds2 =roc_curve(dfsmote["h1n1_vaccine"], y_pred_prob2[:,1])
roc_auc2 = auc(fpr2, tpr2) #Area under Curve 0.82
print(roc_auc2) # 0.82, same as without smote

#ROC Curve
plt.title('ROC Curve for LogReg: liblinear SMOTE')
plt.xlabel('False Positive Rate (1-Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.plot(fpr2, tpr2, label = 'AUC =' +str(roc_auc2))
plt.legend(loc=4) #Location of label
plt.show()

