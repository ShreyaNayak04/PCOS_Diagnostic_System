import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %matplotlib inline

plt.style.use("ggplot")
plt.rcParams['figure.figsize'] = (12, 8)
import seaborn as sns
sns.set(style='whitegrid', color_codes=True)
import warnings
warnings.filterwarnings('ignore')
from sklearn.feature_selection import chi2,f_classif, mutual_info_classif, SelectKBest
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold

df = pd.read_csv('PCOS_clean_data_without_infertility.csv')
df.head(12).T

df.info()

#Get the count
print(df['PCOS (Y/N)'].value_counts())

#Splitting Categorical And Numerical Features

df_cat = df[["Age (yrs)",
            "Pregnant(Y/N)",
             "Cycle(R/I)",
             "Blood Group",
             "Cycle length(days)",
             "No. of aborptions",
             "Weight gain(Y/N)",
             "hair growth(Y/N)",
             "Skin darkening (Y/N)",
             "Hair loss(Y/N)",
             "Pimples(Y/N)",
             "Fast food (Y/N)",
             "Reg.Exercise(Y/N)",

]]

df_cat.shape

df_target = df[["PCOS (Y/N)"]]

df_corr_num = df.drop(df_cat.columns,axis=1)

list(df_corr_num.columns)

df_num = df.drop(df_cat.columns,axis=1)
df_num.drop(["PCOS (Y/N)"], axis=1, inplace= True)

df_corr_num = pd.concat([df_target, df_num], axis=1, sort = False)
#pd.concat([df1, df4], axis=1, sort=False)

df_corr_num.head()

"""Finding Correlations
Kendall's Method
"""

# Numerical Input - Categorical Output
# Kendall's Method
plt.figure(figsize=(32, .8))
sns.heatmap(
    data=df_corr_num.corr('kendall').iloc[:1, 1:],
    annot=True,
    fmt='.0%',
    cmap='coolwarm'


);

#Chi Square**

chi2(df_cat,df_target)[0]

chi2(df_cat,df_target)[1] # p-values

pd.DataFrame.from_records(np.reshape(chi2(df_cat,df_target)[1], (1,-1)), index= list(df_target.columns), columns=list(df_cat.columns)) # p-values

df_corr_chi = pd.DataFrame.from_records(np.reshape(chi2(df_cat,df_target)[0], (1,-1)), index= list(df_target.columns), columns=list(df_cat.columns))

# Categorical Input - Categorical Output
# Chi Square Method
plt.figure(figsize=(20, 1))
sns.heatmap(
    data= df_corr_chi,
    annot=True,
    cmap='coolwarm'
)

df_corr_chi

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
plt.style.use("ggplot")
plt.rcParams['figure.figsize'] = (12, 8)
import seaborn as sns
sns.set(style='whitegrid', color_codes=True)
import warnings
warnings.filterwarnings('ignore')
from sklearn.feature_selection import chi2,f_classif, mutual_info_classif, SelectKBest
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import confusion_matrix

df = pd.read_csv('PCOS_clean_data_without_infertility.csv')
df.head(12)

df.info()

X = df.drop(["PCOS (Y/N)",
             "Weight (Kg)",
            "Pulse rate(bpm)",
            "Height(Cm)",
            "Hb(g/dl)",
             "PRG(ng/mL)",
            "RR (breaths/min)",
            "Marraige Status (Yrs)",
            "Hip(inch)",
            "Waist(inch)",
            "FSH/LH",
            "I   beta-HCG(mIU/mL)",
            "II    beta-HCG(mIU/mL)",
            "TSH (mIU/L)",
            "FSH(mIU/mL)",
            "LH(mIU/mL)",
            "Waist:Hip Ratio",
            "PRL(ng/mL)",
            "BP _Diastolic (mmHg)",
            "BP _Systolic (mmHg)",
            "RBS(mg/dl)",
           "AMH(ng/mL)",
           "Vit D3 (ng/mL)",
           "Follicle No. (L)",
           "Follicle No. (R)",
           "Avg. F size (L) (mm)",
           "Avg. F size (R) (mm)",
           "Endometrium (mm)"

            ],axis=1)

X

y = df[["PCOS (Y/N)"]]

def test_results(model, X_test, y_test):
    from sklearn.metrics import confusion_matrix
    y_pred = model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    accuracy = (tp + tn)/(tp + fp + tn + fn)
    print("Accuracy: ", '{:.2f}'.format(accuracy * 100))
    print("True Negative:", tn)
    print("True Positve:", tp)
    print("False Positive:", fp)
    print("False Negative:", fn)
    print()
    print("-------------------------------------------------------")
    print("Negative Class Results")
    precision = (tp / (tp + fp))
    recall =  (tp  / (tp + fn))
    f1_score = (2 * (precision * recall) / (precision + recall))
    print("Precision (N): ", '{:.2f}'.format(precision * 100))
    print("Recall (N): ", '{:.2f}'.format(recall * 100))
    print("F1 Score (N):" ,  '{:.2f}'.format(f1_score * 100))
    print()
    print("-------------------------------------------------------")
    print("Positive Class Results")
    precision = (tn / (tn + fn))
    recall =  (tn  / (tn + fp))
    f1_score = (2 * (precision * recall) / (precision + recall))
    print("Precision (P): ", '{:.2f}'.format(precision * 100))
    print("Recall (P): ", '{:.2f}'.format(recall * 100))
    print("F1 Score (P):" , '{:.2f}'.format(f1_score * 100))



from yellowbrick.classifier import confusion_matrix

def vis_conf(model, X_test, y_test):
    plt.figure(figsize=(6, 5))
    visualizer = confusion_matrix(
        model,
        X_test, y_test,
        is_fitted=True,
        classes=['Negative', 'Positive']
    )
    visualizer.show();



from imblearn.combine import SMOTEENN

resample = SMOTEENN(sampling_strategy=1/1, random_state =0)
X, y = resample.fit_resample(X, y)

# print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify= y)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler().fit(X_train)

X_train = scaler.transform(X_train)
X_train = pd.DataFrame(X_train)

X_test = scaler.transform(X_test)
X_test = pd.DataFrame(X_test)


# Setting Column Names from dataset
X_train.columns = X.columns
X_test.columns = X.columns

X_train.shape



# from sklearn.ensemble import RandomForestClassifier
# rf = RandomForestClassifier(n_estimators=1500, criterion="entropy", random_state=0)
# rf.fit(X_train, y_train)
# test_results(rf, X_test, y_test)
# vis_conf(rf, X_test, y_test)
# import pickle

# pickle.dump(rf, open('modelfinal1.pkl', 'wb'))


import xgboost as xgb
xgb = xgb.XGBClassifier(max_depth=5, n_estimators=1500, learning_rate=0.3,scale_pos_weight=10,
                            random_state= 0, n_jobs=-1)
sum(cross_val_score(xgb, X, y, cv=10))/10
xgb = xgb.fit(X_train, y_train)
test_results(xgb, X_test, y_test)
vis_conf(xgb, X_test, y_test)
import pickle
pickle.dump(xgb, open('modelfinal1.pkl', 'wb'))


# from catboost import CatBoostClassifier
# import warnings
# warnings.filterwarnings('ignore')
# cv_result = []
# cat_clf = CatBoostClassifier()
# cat_clf.fit(X_train,y_train)
# acc_cat_clf_train = round(cat_clf.score(X_train, y_train)*100,2)
# acc_cat_clf_test = round(cat_clf.score(X_test,y_test)*100,2)
# cv_result.append(acc_cat_clf_train)
# print("Training Accuracy: % {}".format(acc_cat_clf_train))
# print("Testing Accuracy: % {}".format(acc_cat_clf_test))
# import pickle
# pickle.dump(cat_clf, open('modelfinal1.pkl', 'wb'))



# import lightgbm as lgb
# from sklearn.metrics import accuracy_score
# # Create LightGBM dataset
# train_data = lgb.Dataset(X_train, label=y_train)
# test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# params = {
#     'objective': 'binary',
#     'metric': 'binary_logloss',
#     'num_leaves': 31,
#     'learning_rate': 0.05,
#     'feature_fraction': 0.9,
#     'bagging_fraction': 0.8,
#     'bagging_freq': 5,
#     'verbose': 0
# }


# # Train the LightGBM model
# num_round = 100
# bst = lgb.train(params, train_data, num_round, valid_sets=[test_data])

# # Make predictions on the testing set
# y_pred = (bst.predict(X_test) > 0.5).astype(int)

# # Calculate accuracy
# accuracy = accuracy_score(y_test, y_pred)*100
# print(accuracy)

















